// BSD 3-Clause License
//
// Copyright (c) 2019, Jiong Tao, Bailin Deng, Yue Peng
// Modified work Copyright 2020, Yue Peng
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include "GeodesicDistanceSolver.h"
#include "surface_mesh/IO.h"
#include "OMPHelper.h"
#include <iostream>
#include <utility>
#include <limits>
#include <fstream>
#include <iostream>
#include <iomanip>

GeodesicDistanceSolver::GeodesicDistanceSolver()
    : model_scaling_factor(1.0),
      bfs_laplacian_coef(NULL),
      need_compute_residual_norms(false),
      prev_SG(NULL),
      current_SG(NULL),
      n_vertices(0),
      n_faces(0),
      n_edges(0),
      n_interior_edges(0),
      iter_num(0),
      aa(NULL),
      primal_residual_sqr_norm(0),
      dual_residual_sqr_norm(0),
      primal_residual_sqr_norm_threshold(0),
      dual_residual_sqr_norm_threshold(0),
      output_progress(false),
      optimization_converge(false),
      optimization_end(false) {
}

const Eigen::VectorXd& GeodesicDistanceSolver::get_distance_values() {
    return geod_dist_values;
}

bool GeodesicDistanceSolver::solve(const char *mesh_file,
                                   const Parameters& para) {
    param = para;

    std::cout << "Reading triangle mesh......" << std::endl;

    if (!load_input(mesh_file)) {
        return false;
    }

    normalize_mesh();

    std::cout << "Initialize BFS path......" << std::endl;

    Timer timer;
    Timer::EventID start = timer.get_time();

    // Precompute breadth-first propagation order
    init_bfs_paths();
    std::cout << "Gauss-Seidel initilization of gradients......" << std::endl;

    Timer::EventID before_GS = timer.get_time();

    gauss_seidel_init_gradients();

    Timer::EventID before_ADMM = timer.get_time();
    std::cout << "ADMM solver for integrable gradients......" << std::endl;

    prepare_integrate_geodesic_distance();

    if (param.Anderson_m > 0 && param.acceleration_type == Parameters::DR)
        compute_integrable_gradients_DR();
    else
        compute_integrable_gradients();

    Timer::EventID after_ADMM = timer.get_time();
    std::cout << "Recovery of geodesic distance......" << std::endl;

    integrate_geodesic_distance();

    Timer::EventID end = timer.get_time();

    std::cout << std::endl;
    std::cout << "====== Timing ======" << std::endl;
    std::cout << "Pre-computation of BFS paths: " <<  timer.elapsed_time(start, before_GS) << " seconds" << std::endl;
    std::cout << "Gauss-Seidel initialization of gradients: " << timer.elapsed_time(before_GS, before_ADMM) << " seconds" << std::endl;
    std::cout << "ADMM solver for integrable gradients: " << timer.elapsed_time(before_ADMM, after_ADMM) << " seconds" << std::endl;
    std::cout << "Integration of gradients: " << timer.elapsed_time(after_ADMM, end) << " seconds" <<  std::endl;
    std::cout << "Total time: " << timer.elapsed_time(start, end) << " seconds" << std::endl;

    // Modified features by
#ifdef CERR_TIME_RAM
    std::cerr << timer.elapsed_time(start, before_GS) << "\t";
    std::cerr << timer.elapsed_time(before_GS, before_ADMM) << "\t";
    std::cerr << timer.elapsed_time(before_ADMM, after_ADMM) << "\t";
    std::cerr << timer.elapsed_time(after_ADMM, end) <<  "\t";
    std::cerr << timer.elapsed_time(start, end) << "\t";
#endif
    return true;
}

void GeodesicDistanceSolver::init_bfs_paths() {
    bfs_vertex_list.resize(n_vertices);
    bfs_vertex_list.fill(-1);
    transition_halfedge_idx.setConstant(n_vertices, -1);

    bfs_laplacian_coef_addr.resize(n_vertices + 1);
    bfs_laplacian_coef_addr(0) = 0;

    // Store source vertices as the first layer
    std::vector<bool> visited(n_vertices, false);
    std::vector<int> front1 = param.source_vertices, front2;
    std::vector<int> *current_front = &front1, *next_front = &front2;

    int n_sources = param.source_vertices.size();

    std::vector<int> bfs_segment_addr_vec;
    bfs_segment_addr_vec.push_back(0);
    bfs_segment_addr_vec.push_back(n_sources);

    int id = 0;
    for (; id < n_sources; ++id) {
        int current_source_vtx = param.source_vertices[id];
        visited[current_source_vtx] = true;
        bfs_vertex_list(id) = current_source_vtx;
        bfs_laplacian_coef_addr(id + 1) = bfs_laplacian_coef_addr(id)
                + mesh.valence(MeshType::Vertex(current_source_vtx)) + 1;
    }

    while (!current_front->empty()) {

        next_front->clear();

        for (int k = 0; k < static_cast<int>(current_front->size()); ++k) {
            MeshType::Vertex vh(current_front->at(k));
            MeshType::Halfedge_around_vertex_circulator vhc, vhc_end;
            vhc = vhc_end = mesh.halfedges(vh);

            do {
                MeshType::Halfedge heh = *vhc;
                MeshType::Vertex next_vh = mesh.to_vertex(heh);
                int next_v = next_vh.idx();

                if (!visited[next_v]) {
                    next_front->push_back(next_v);
                    bfs_vertex_list(id) = next_v;
                    // Each segment stores the weights for neighbors and the current vertex for GS update
                    bfs_laplacian_coef_addr(id + 1) = bfs_laplacian_coef_addr(id)
                            + mesh.valence(next_vh) + 1;
                    transition_halfedge_idx(id) = heh.idx();
                    id++;
                }

                visited[next_v] = true;

            } while (++vhc != vhc_end);
        }

        bfs_segment_addr_vec.push_back(
                    bfs_segment_addr_vec.back() + next_front->size());
        std::swap(current_front, next_front);
    }

    bfs_segment_addr = Eigen::Map<Eigen::VectorXi>(bfs_segment_addr_vec.data(),
                                                   bfs_segment_addr_vec.size());
}

bool GeodesicDistanceSolver::load_input(const char* mesh_file) {
    if (!surface_mesh::read_mesh(mesh, mesh_file)) {
        std::cerr << "Error: unable to read input mesh from the file " << mesh_file
                  << std::endl;
        return false;
    }

    mesh.free_memory();  // Free unused memory

    n_vertices = mesh.n_vertices();
    n_faces = mesh.n_faces();
    n_edges = mesh.n_edges();

    if(n_vertices == 0  || n_faces == 0 || n_edges == 0){
        std::cerr << "Error: zero mesh element count " << std::endl;
        return false;
    }

    for (int i = 0; i < static_cast<int>(param.source_vertices.size()); ++i) {
        if (param.source_vertices[i] < 0
                || param.source_vertices[i] >= n_vertices) {
            std::cerr << "Error: invalid source vertex index "
                      << param.source_vertices[i] << std::endl;
            return false;
        }
    }

    return true;
}

void GeodesicDistanceSolver::normalize_mesh()
{
    std::vector<surface_mesh::Point> &pos = mesh.points();

    surface_mesh::Point min_coord = pos.front(), max_coord = pos.front();
    std::vector<surface_mesh::Point>::iterator iter = pos.begin(), iter_end = pos.end();

    for(++ iter; iter != iter_end; ++ iter){
        surface_mesh::Point &coord = *iter;
        min_coord.minimize(coord);
        max_coord.maximize(coord);
    }

    model_scaling_factor = surface_mesh::norm(max_coord - min_coord);
    surface_mesh::Point center_pos = (min_coord + max_coord) * 0.5;

    for(iter = pos.begin(); iter != iter_end; ++ iter){
        surface_mesh::Point &coord = *iter;
        coord -= center_pos;
        coord /= model_scaling_factor;
    }
}


/*
void GeodesicDistanceSolver::gauss_seidel_init_gradients()
{
    edge_vector.resize(3, n_edges);
    DenseVector edge_sqr_length;
    edge_sqr_length.setZero(n_edges);

    face_area.resize(n_faces);

    int n_halfedges = mesh.n_halfedges();
    DenseVector halfedge_halfcot;  // Collect the half cotan value for edge halfedge, to be used for computing cotan Laplacian weights
    halfedge_halfcot.setZero(n_halfedges);

    double step_length = 0;

    typedef long double HeatScalar;
    typedef Eigen::Matrix<HeatScalar, Eigen::Dynamic, 1> VectorHS;
    typedef Eigen::Matrix<HeatScalar, 3, 1> Vector3HS;
    typedef Eigen::Matrix<HeatScalar, 3, 3> Matrix3HS;
    //VectorHS d1, d2;  // Heat values;
    VectorHS source_d;
    VectorHS current_d, prev_d;
    VectorHS temp_d;
    VectorHS vertex_area;
    int gs_iter = 0;
    int segment_count = 0;
    int n_segments = 0;
    int segment_begin_addr = 0, segment_end_addr = 0;
    bool end_gs_loop = false;
    bool reset_iter = true;
    HeatScalar eps = 0;
    Eigen::VectorXd diag_scales;


    OMP_PARALLEL
    {
        // Compute Laplacian weights
        OMP_FOR
                for (int i = 0; i < n_edges; ++i) {
            // Precompute edge vectors and squared edge length,
            // to be used later for computing cotan weights and areas
            MeshType::Halfedge heh = mesh.halfedge(MeshType::Edge(i), 0);
            Eigen::Vector3d edge_vec = to_eigen_vec3d(
                        mesh.position(mesh.to_vertex(heh))
                        - mesh.position(mesh.from_vertex(heh)));
            double l2 = edge_vec.squaredNorm();
            edge_vector.col(i) = edge_vec;
            edge_sqr_length(i) = l2;
        }

        OMP_SINGLE
        {
            // Compute heat flow step size
            double h = edge_sqr_length.array().sqrt().mean();
            step_length = h * h;
            std::cout << "step_length = " << step_length << std::endl;
        }

        OMP_FOR
                for (int i = 0; i < n_faces; ++i) {
            // Compute face areas and half-cotan weights for halfedges
            Eigen::Vector3i fh_idx, fe_idx;
            Eigen::Vector3d edge_l2;
            int k = 0;

            MeshType::Halfedge_around_face_circulator fhc, fhc_end;
            fhc = fhc_end = mesh.halfedges(MeshType::Face(i));
            do {
                MeshType::Halfedge heh = *fhc;
                fh_idx(k) = heh.idx();
                fe_idx(k) = mesh.edge(heh).idx();
                edge_l2(k) = edge_sqr_length(fe_idx(k));
                k++;
            } while (++fhc != fhc_end);

            double area = edge_vector.col(fe_idx(0)).cross(edge_vector.col(fe_idx(1))).norm() * 0.5;
            for (int j = 0; j < 3; ++j) {
                halfedge_halfcot(fh_idx(j)) = 0.125
                        * (edge_l2((j + 1) % 3) + edge_l2((j + 2) % 3) - edge_l2(j)) / area;
            }

            face_area(i) = area;
        }

        OMP_SINGLE
        {
            double avg_vtx_area = face_area.sum() / double(n_vertices);
            step_length /= avg_vtx_area;
            face_area /= avg_vtx_area;

            // Allocate arrays for storing relevant vertices and weights for Laplacian operator at each vertex
            edge_sqr_length.resize(0);
            int n_laplacian_vertices = n_edges * 2 + n_vertices;
            bfs_laplacian_coef = new std::pair<int, double>[n_laplacian_vertices];
            vertex_area.setZero(n_vertices);
            diag_scales.setZero(n_vertices);
        }


        OMP_FOR
                for (int i = 0; i < n_vertices; ++i)
        {
            // Compute and store vertex indices and weights for Laplacian operators
            int start_addr = bfs_laplacian_coef_addr(i), end_addr = bfs_laplacian_coef_addr(i + 1);

            DenseVector weights;
            IndexVector vtx_idx;
            int n = end_addr - start_addr;
            weights.setZero(n);
            vtx_idx.setZero(n);

            int v_idx = bfs_vertex_list(i);
            MeshType::Vertex vh(v_idx);
            int k = 0;
            MeshType::Halfedge_around_vertex_circulator vhc, vhc_end;
            vhc = vhc_end = mesh.halfedges(vh);

            do {
                MeshType::Halfedge heh = *vhc;
                double w = halfedge_halfcot(heh.idx()) + halfedge_halfcot(mesh.opposite_halfedge(heh).idx());
                vtx_idx(k) = mesh.to_vertex(heh).idx();
                weights(k) = w;
                k++;
            } while (++vhc != vhc_end);

            vtx_idx(k) = v_idx;
            weights(k) = weights.head(k).sum();  // Store the sum of neighbor weights, to be used for Gauss-Seidel update
            weights *= step_length;

            double A = 0;
            MeshType::Face_around_vertex_circulator vfc, vfc_end;
            vfc = vfc_end = mesh.faces(vh);
            do {
                A += face_area((*vfc).idx());
            } while (++vfc != vfc_end);

            double vertex_A = A / 3.0;
            vertex_area(v_idx) = vertex_A;
            weights(k) += vertex_A;
            diag_scales(v_idx) = std::fabs(weights(k));

            for(int j = start_addr; j < end_addr; ++ j){
                bfs_laplacian_coef[j] = std::pair<int, double>(vtx_idx(j-start_addr), weights(j-start_addr));
            }
        }

        OMP_FOR
                for (int i = 0; i < n_vertices; ++i)
        {
            int start_addr = bfs_laplacian_coef_addr(i), end_addr = bfs_laplacian_coef_addr(i + 1);
            for(int j = start_addr; j < end_addr; ++ j){
                bfs_laplacian_coef[j].second /= diag_scales(bfs_laplacian_coef[j].first);
            }
        }

        OMP_SINGLE
        {
            // Set up heat value arrays
            halfedge_halfcot.resize(0);
            current_d.setZero(n_vertices);
            prev_d.setZero(n_vertices);
            int n_sources = param.source_vertices.size();
            HeatScalar init_source_val = std::sqrt(double(n_vertices) / n_sources);
            source_d.setConstant(n_sources, init_source_val);
            for (int i = 0; i < n_sources; ++i) {
                current_d(param.source_vertices[i]) = source_d(i) * diag_scales(param.source_vertices[i]);
            }
            n_segments = bfs_segment_addr.size() - 1;
            int buffer_size = (Eigen::Map<IndexVector>(&(bfs_segment_addr[1]), n_segments)
                    - Eigen::Map<IndexVector>(&(bfs_segment_addr[0]), n_segments)).maxCoeff();
            temp_d.setZero(buffer_size);

            vertex_area /= vertex_area.mean();
            eps = double(n_vertices) * param.heat_solver_eps;
        }

        // Rescale vertex areas to be used later for checking solution change between iteration
        OMP_FOR
                for(int i = 0; i < n_vertices; ++ i){
            vertex_area(i) /= diag_scales(i);
        }

        while (!end_gs_loop)
        {
            // Gauss-Seidel update of heat values in breadth-first order
            OMP_SINGLE
            {
                if(reset_iter){
                    prev_d = current_d;
                }
                segment_begin_addr = bfs_segment_addr(segment_count);
                segment_end_addr = bfs_segment_addr(segment_count + 1);
            }

            OMP_FOR
                    for (int i = segment_begin_addr; i < segment_end_addr; ++i) {
                int lap_coef_begin_addr = bfs_laplacian_coef_addr(i);
                int lap_coef_end_addr = bfs_laplacian_coef_addr(i + 1);

                HeatScalar new_heat_value = 0;
                if (segment_count == 0) {  // Check whether the current vertex is a source
                    new_heat_value += source_d(i);
                }

                for (int j = lap_coef_begin_addr; j < lap_coef_end_addr - 1; ++j) {
                    std::pair<int, double> &coef = bfs_laplacian_coef[j];
                    new_heat_value += current_d(coef.first) * coef.second;
                }

                temp_d(i - segment_begin_addr) = new_heat_value / bfs_laplacian_coef[lap_coef_end_addr - 1].second;
            }

            OMP_FOR
                    for (int i = segment_begin_addr; i < segment_end_addr; ++i) {
                current_d(bfs_vertex_list(i)) = temp_d(i - segment_begin_addr);
            }

            OMP_SINGLE
            {
                segment_count++;
                reset_iter = (segment_count == n_segments);
                if (reset_iter) {
                    gs_iter++;
                    segment_count = 0;
                }
                end_gs_loop = gs_iter >= param.heat_solver_max_iter;

                if (end_gs_loop || (reset_iter && (gs_iter % param.heat_solver_convergence_check_frequency == 0))) {
                    HeatScalar disp_sum = vertex_area.dot((current_d - prev_d).array().abs().matrix());
                    std::cout << "Gauss-Seidel iteration " << gs_iter
                              << ", total heat change: " << disp_sum << ", threshold:"
                              << eps << std::endl;
                    if (disp_sum <= eps) {
                        end_gs_loop = true;
                    }
                }
            }
        }

        // Generate final solution
        OMP_FOR
                for(int i = 0; i < n_vertices; ++ i){
            current_d(i) /= diag_scales(i);
        }

        OMP_SINGLE
        {
            diag_scales.resize(0);
            prev_d.resize(0);
            temp_d.resize(0);
            vertex_area.resize(0);
            delete [] bfs_laplacian_coef;
            bfs_laplacian_coef_addr.resize(0);
            init_grad.resize(3, n_faces);
        }

        // Compute initial gradient
        OMP_FOR
                for (int i = 0; i < n_faces; ++i) {
            Matrix3HS edge_vecs;
            Vector3HS heat_vals;
            int k = 0;

            MeshType::Halfedge_around_face_circulator fhc, fhc_end;
            fhc = fhc_end = mesh.halfedges(MeshType::Face(i));

            do {
                MeshType::Halfedge heh = *fhc;
                MeshType::Edge eh = mesh.edge(heh);
                Eigen::Vector3d current_edge = edge_vector.col(eh.idx());
                if (mesh.halfedge(eh, 0) != heh) {
                    current_edge *= -1;
                }

                edge_vecs(0, k) = HeatScalar(current_edge[0]);
                edge_vecs(1, k) = HeatScalar(current_edge[1]);
                edge_vecs(2, k) = HeatScalar(current_edge[2]);
                heat_vals(k) = current_d(mesh.to_vertex(heh).idx());
                ++k;

            } while (++fhc != fhc_end);

            heat_vals.normalize();
            edge_vecs.normalize();

            Vector3HS N = edge_vecs.col(0).cross(edge_vecs.col(1)).normalized();
            Vector3HS V = edge_vecs.col(0) * heat_vals(1)
                    + edge_vecs.col(1) * heat_vals(2) + edge_vecs.col(2) * heat_vals(0);
            Vector3HS grad_vec = V.cross(N).normalized();
            init_grad(0, i) = grad_vec(0);
            init_grad(1, i) = grad_vec(1);
            init_grad(2, i) = grad_vec(2);
        }
    }
}
*/


void GeodesicDistanceSolver::gauss_seidel_init_gradients()
{
    edge_vector.resize(3, n_edges);
    DenseVector edge_sqr_length;
    edge_sqr_length.setZero(n_edges);

    face_area.resize(n_faces);

    int n_halfedges = mesh.n_halfedges();
    DenseVector halfedge_halfcot;  // Collect the half cotan value for edge halfedge, to be used for computing cotan Laplacian weights
    halfedge_halfcot.setZero(n_halfedges);

    double step_length = 0;
    HeatScalar init_source_val = 1;
    VectorHS current_d;
    VectorHS temp_d;
    VectorHS vertex_area;
    int gs_iter = 0;
    int segment_count = 0;
    int n_segments = 0;
    int segment_begin_addr = 0, segment_end_addr = 0;
    bool end_gs_loop = false;
    bool reset_iter = true;
    bool need_check_residual = false;
    HeatScalar eps = 0;
    VectorHS heatflow_residuals;

    OMP_PARALLEL
    {
        int threadID = omp_get_thread_num();
        printf("Thread %d reporting\n", threadID);
        // Compute Laplacian weights
        OMP_FOR
                for (int i = 0; i < n_edges; ++i) {
            // Precompute edge vectors and squared edge length,
            // to be used later for computing cotan weights and areas
            MeshType::Halfedge heh = mesh.halfedge(MeshType::Edge(i), 0);
            Eigen::Vector3d edge_vec = to_eigen_vec3d(
                        mesh.position(mesh.to_vertex(heh))
                        - mesh.position(mesh.from_vertex(heh)));
            double l2 = edge_vec.squaredNorm();
            edge_vector.col(i) = edge_vec;
            edge_sqr_length(i) = l2;
        }

        OMP_SINGLE
        {
            // Compute heat flow step size
            double h = edge_sqr_length.array().sqrt().mean();

            // Modified features by
            step_length = h * h * param.heat_solver_time;
        }

        OMP_FOR
                for (int i = 0; i < n_faces; ++i) {
            // Compute face areas and half-cotan weights for halfedges
            Eigen::Vector3i fh_idx, fe_idx;
            Eigen::Vector3d edge_l2;
            int k = 0;

            MeshType::Halfedge_around_face_circulator fhc, fhc_end;
            fhc = fhc_end = mesh.halfedges(MeshType::Face(i));
            do {
                MeshType::Halfedge heh = *fhc;
                fh_idx(k) = heh.idx();
                fe_idx(k) = mesh.edge(heh).idx();
                edge_l2(k) = edge_sqr_length(fe_idx(k));
                k++;
            } while (++fhc != fhc_end);

            double area = edge_vector.col(fe_idx(0)).cross(edge_vector.col(fe_idx(1)))
                    .norm() * 0.5;
            for (int j = 0; j < 3; ++j) {
                halfedge_halfcot(fh_idx(j)) = 0.125
                        * (edge_l2((j + 1) % 3) + edge_l2((j + 2) % 3) - edge_l2(j)) / area;
            }

            face_area(i) = area;
        }

        OMP_SINGLE
        {
            // Allocate arrays for storing relevant vertices and weights for Laplacian operator at each vertex
            edge_sqr_length.resize(0);
            int n_laplacian_vertices = n_edges * 2 + n_vertices;
            bfs_laplacian_coef = new std::pair<int, double>[n_laplacian_vertices];
            vertex_area.setZero(n_vertices);
        }


        OMP_FOR
                for (int i = 0; i < n_vertices; ++i)
        {
            // Compute and store vertex indices and weights for Laplacian operators
            int start_addr = bfs_laplacian_coef_addr(i), end_addr = bfs_laplacian_coef_addr(i + 1);

            DenseVector weights;
            IndexVector vtx_idx;
            int n = end_addr - start_addr;
            weights.setZero(n);
            vtx_idx.setZero(n);

            int v_idx = bfs_vertex_list(i);
            MeshType::Vertex vh(v_idx);
            int k = 0;
            MeshType::Halfedge_around_vertex_circulator vhc, vhc_end;
            vhc = vhc_end = mesh.halfedges(vh);

            do {
                MeshType::Halfedge heh = *vhc;
                double w = halfedge_halfcot(heh.idx()) + halfedge_halfcot(mesh.opposite_halfedge(heh).idx());
                vtx_idx(k) = mesh.to_vertex(heh).idx();
                weights(k) = w;
                k++;
            } while (++vhc != vhc_end);

            vtx_idx(k) = v_idx;
            weights(k) = weights.head(k).sum();  // Store the sum of neighbor weights, to be used for Gauss-Seidel update
            weights *= step_length;

            double A = 0;
            MeshType::Face_around_vertex_circulator vfc, vfc_end;
            vfc = vfc_end = mesh.faces(vh);
            do {
                A += face_area((*vfc).idx());
            } while (++vfc != vfc_end);

            double vertex_A = A / 3.0;
            vertex_area(v_idx) = vertex_A;
            weights(k) += vertex_A;

            for(int j = start_addr; j < end_addr; ++ j){
                bfs_laplacian_coef[j] = std::pair<int, double>(vtx_idx(j-start_addr), weights(j-start_addr));
            }
        }

        OMP_SINGLE
        {
            // Set up heat value arrays
            halfedge_halfcot.resize(0);

            int n_sources = param.source_vertices.size();
            HeatScalar total_source_area = 0;
            for(int i = 0; i < n_sources; ++ i){
                total_source_area += vertex_area(param.source_vertices[i]);
            }
            init_source_val = std::sqrt(std::min(HeatScalar(n_vertices)/HeatScalar(n_sources), vertex_area.sum()/total_source_area));
            vertex_area.resize(0);

            current_d.setZero(n_vertices);
            for (int i = 0; i < n_sources; ++i) {
                current_d(param.source_vertices[i]) = init_source_val;
            }

            n_segments = bfs_segment_addr.size() - 1;
            int buffer_size = (Eigen::Map<IndexVector>(&(bfs_segment_addr[1]), n_segments)
                    - Eigen::Map<IndexVector>(&(bfs_segment_addr[0]), n_segments)).maxCoeff();
            temp_d.setZero(buffer_size);

            heatflow_residuals.setZero(n_vertices);
        }

        compute_heatflow_residual(current_d, init_source_val, heatflow_residuals);

        OMP_SINGLE
        {
            // Rescale heat source values to make the initial residual norm close to 1
            HeatScalar init_residual_norm = heatflow_residuals.norm();
            eps = std::max(HeatScalar(1e-16), init_residual_norm * HeatScalar(param.heat_solver_eps));
            std::cout << "Initial residual: " << init_residual_norm <<", threshold: " << eps << std::endl;
        }

        while (!end_gs_loop)
        {
            // Gauss-Seidel update of heat values in breadth-first order
            OMP_SINGLE
            {
                segment_begin_addr = bfs_segment_addr(segment_count);
                segment_end_addr = bfs_segment_addr(segment_count + 1);
            }

            OMP_FOR
                    for (int i = segment_begin_addr; i < segment_end_addr; ++i) {
                int lap_coef_begin_addr = bfs_laplacian_coef_addr(i);
                int lap_coef_end_addr = bfs_laplacian_coef_addr(i + 1);

                HeatScalar new_heat_value = 0;
                if (segment_count == 0) {  // Check whether the current vertex is a source
                    new_heat_value += init_source_val;
                }

                for (int j = lap_coef_begin_addr; j < lap_coef_end_addr - 1; ++j) {
                    std::pair<int, double> &coef = bfs_laplacian_coef[j];
                    new_heat_value += current_d(coef.first) * coef.second;
                }

                temp_d(i - segment_begin_addr) = new_heat_value / bfs_laplacian_coef[lap_coef_end_addr - 1].second;
            }

            OMP_FOR
                    for (int i = segment_begin_addr; i < segment_end_addr; ++i) {
                current_d(bfs_vertex_list(i)) = temp_d(i - segment_begin_addr);
            }

            OMP_SINGLE
            {
                segment_count++;
                reset_iter = (segment_count == n_segments);
                if (reset_iter) {
                    gs_iter++;
                    segment_count = 0;
                }

                end_gs_loop = gs_iter >= param.heat_solver_max_iter;
                need_check_residual = end_gs_loop || (reset_iter && (gs_iter % param.heat_solver_convergence_check_frequency == 0));
            }

            if(need_check_residual)
            {
                compute_heatflow_residual(current_d, init_source_val, heatflow_residuals);

                OMP_SINGLE
                {
                    HeatScalar residual_norm = heatflow_residuals.norm();
                    std::cout << "Gauss-Seidel iteration " << gs_iter
                              << ", current residual: " << residual_norm
                              << ", threshold: " << eps << std::endl;

                    if(residual_norm <= eps){
                        end_gs_loop = true;
                    }
                }
            }
        }

        OMP_SINGLE
        {
            temp_d.resize(0);
            heatflow_residuals.resize(0);
            vertex_area.resize(0);
            delete [] bfs_laplacian_coef;
            bfs_laplacian_coef_addr.resize(0);
            init_grad.resize(3, n_faces);
        }

        // Compute initial gradient
        OMP_FOR
                for (int i = 0; i < n_faces; ++i) {
            Matrix3HS edge_vecs;
            Vector3HS heat_vals;
            int k = 0;

            MeshType::Halfedge_around_face_circulator fhc, fhc_end;
            fhc = fhc_end = mesh.halfedges(MeshType::Face(i));

            do {
                MeshType::Halfedge heh = *fhc;
                MeshType::Edge eh = mesh.edge(heh);
                Eigen::Vector3d current_edge = edge_vector.col(eh.idx());
                if (mesh.halfedge(eh, 0) != heh) {
                    current_edge *= -1;
                }

                edge_vecs(0, k) = HeatScalar(current_edge[0]);
                edge_vecs(1, k) = HeatScalar(current_edge[1]);
                edge_vecs(2, k) = HeatScalar(current_edge[2]);
                heat_vals(k) = current_d(mesh.to_vertex(heh).idx());
                ++k;

            } while (++fhc != fhc_end);

            heat_vals.normalize();
            edge_vecs.normalize();

            Vector3HS N = edge_vecs.col(0).cross(edge_vecs.col(1)).normalized();
            Vector3HS V = edge_vecs.col(0) * heat_vals(1)
                    + edge_vecs.col(1) * heat_vals(2) + edge_vecs.col(2) * heat_vals(0);
            Vector3HS grad_vec = V.cross(N).normalized();
            init_grad(0, i) = grad_vec(0);
            init_grad(1, i) = grad_vec(1);
            init_grad(2, i) = grad_vec(2);
        }
    }

}

void GeodesicDistanceSolver::compute_heatflow_residual(const VectorHS &heat_values, HeatScalar init_source_val, VectorHS &residuals)
{
    OMP_FOR
            for (int i = 0; i < n_vertices; ++ i) {
        int lap_coef_begin_addr = bfs_laplacian_coef_addr(i);
        int lap_coef_end_addr = bfs_laplacian_coef_addr(i + 1);

        HeatScalar res = 0;
        if (i < static_cast<int>(param.source_vertices.size())) {  // Check whether the current vertex is a source
            res += init_source_val;
        }

        for (int j = lap_coef_begin_addr; j < lap_coef_end_addr; ++j) {
            std::pair<int, double> &coef = bfs_laplacian_coef[j];
            res += heat_values(coef.first) * coef.second * ((j == (lap_coef_end_addr-1)) ? (-1) : 1);
        }

        residuals(i) = res;
    }
}


//void GeodesicDistanceSolver::compute_integrable_gradients() {
//    optimization_end = false;
//    iter_num = 0;
//    bool accel = param.Anderson_m > 0;
//    bool accept = false, reset = false;
//    double prev_residual = std::numeric_limits<double>::max();
//    double current_residual = 0;
//    default_G_ = G;
//    default_D_ = D;
//    new_G_ = G;
//    new_D_ = D;

//    std::cout << "current_SG dim = " << current_SG->size() << std::endl;

//    if (accel) {
//      aa = new AndersonAcceleration(param.Anderson_m,
//                                    G.size() + D.size(),
//                                    G.size() + D.size());
//      aa->init(G, D);
//    }

//    Timer timer_;
//    Timer::EventID start_time_;
//    int count_reject = 0;
//    double AA_reset_time = 0.0, AA_compute_time = 0.0;
//    Timer::EventID AA_reset_time_start;


//    while (!optimization_end) {
//        OMP_SINGLE
//        {
//            if(accel && reset)
//            {
//                AA_reset_time_start = timer_.get_time();
//            }
//        }

//        OMP_PARALLEL
//        {
//            OMP_FOR
//            for (int i = 0; i < n_interior_edges; ++ i) {
//                prev_SG->col(2 * i) = G.col(S(0, i));
//                prev_SG->col(2 * i + 1) = G.col(S(1, i));
//            }

//            update_Y();
//            update_G();
//            update_dual_variables();
//        }

//        OMP_SINGLE
//        {
//            iter_num++;
//            optimization_converge = need_compute_residual_norms
//                    && (primal_residual_sqr_norm <= primal_residual_sqr_norm_threshold
//                        && dual_residual_sqr_norm <= dual_residual_sqr_norm_threshold);
//            optimization_end = optimization_converge || iter_num >= param.grad_solver_max_iter ;
//            output_progress = need_compute_residual_norms && (iter_num % param.grad_solver_output_frequency == 0);

//            if (optimization_converge) {
//                std::cout << "Solver converged." << std::endl;
//            } else if (optimization_end) {
//                std::cout << "Maximum number of iterations reached." << std::endl;
//            }

//            if (output_progress || optimization_end) {
//                std::cout << "Iteration " << iter_num << ":" << std::endl;
//                std::cout << "Primal residual squared norm: " << primal_residual_sqr_norm
//                          << ",  threshold:" << primal_residual_sqr_norm_threshold
//                          << std::endl;
//                std::cout << "Dual residual squared norm: " << dual_residual_sqr_norm
//                          << ",  threshold:" << dual_residual_sqr_norm_threshold
//                          << std::endl;

//                if (optimization_end)
//                    std::cerr << iter_num << '\t';
//            }

////            // Modified features by
////#ifdef CERR_RESIDUALS
////            //        std::cerr << iter_num << "\t" << primal_residual_sqr_norm << "\t" << dual_residual_sqr_norm << std::endl;
////#endif

//            if (!optimization_end)
//            {
//                current_residual = (Y - (*current_SG)).colwise().squaredNorm().dot(Y_area)
//                        + ((*prev_SG) - (*current_SG)).colwise().squaredNorm().dot(Y_area_squared);
//                //std::cout << "current_residual = " << current_residual << std::endl;

//                if(accel && reset)
//                {
//                    AA_reset_time += timer_.elapsed_time(AA_reset_time_start, timer_.get_time());
//                }

//                accept = (!accel) || reset || current_residual < prev_residual;

//                if (accept) {
//                    if (reset) {
//                        //std::cout << ", AA was reset";
//                        count_reject++;
//                    }
//                    //std::cout << std::endl;

//                    prev_residual = current_residual;
//                    reset = false;

//                    function_values_.push_back(current_residual);
//                    Timer::EventID iter_time_ = timer_.get_time();
//                    elapsed_time_.push_back(timer_.elapsed_time(start_time_, iter_time_));

//                    if (accel) {
//                        default_G_ = new_G_;
//                        default_D_ = new_D_;

//                        Timer::EventID AA_time = timer_.get_time();
//                        aa->compute(new_G_, new_D_, G, D);
//                        AA_compute_time += timer_.elapsed_time(AA_time, timer_.get_time());

//                    } else {
//                        G = new_G_;
//                        D = new_D_;
//                    }
//                } else {
//                    G = default_G_;
//                    D = default_D_;
//                    reset = true;

//                    //if (accel) {
//                        aa->reset(G, D);
//                        //std::cout << "iter = " << iter_num << ", reject." << std::endl;
//                    //}
//                }
//            }

//        }

////        OMP_SINGLE
////        {
////            std::swap(current_SG, prev_SG);
////        }
//    }

//    std::cout << "reject number = " << count_reject << std::endl;
//    double solving_time = timer_.elapsed_time(start_time_, timer_.get_time());
//    std::cout << "Solving time: " << solving_time << " secs." << std::endl;
//    std::cout << "AA reset time: " << AA_reset_time << " secs." << std::endl;
//    std::cout << "AA compute time: " << AA_compute_time << " secs." << std::endl;

//    save(accel, param.Anderson_m);

//    if (aa) {
//        delete aa;
//    }
//}

void GeodesicDistanceSolver::compute_integrable_gradients() {
    optimization_end = false;
    iter_num = 0;
    bool accel = param.Anderson_m > 0;
    bool accept = false, reset = false;
    double prev_residual = std::numeric_limits<double>::max();
    double current_residual = 0;
    default_Y_ = Y;
    default_D_ = D;
    new_Y_ = Y;
    new_D_ = D;

    std::cout << "current_SG dim = " << current_SG->size() << std::endl;

    if (accel) {
        aa = new AndersonAcceleration(param.Anderson_m,
                                      Y.size() + D.size(),
                                      Y.size() + D.size());
        aa->init(Y, D);
    }

    Timer timer_;
    Timer::EventID start_time_;
    int count_reject = 0;
    double AA_reset_time = 0.0, AA_compute_time = 0.0, total_time = 0.0;
    Timer::EventID AA_reset_time_start;


    OMP_PARALLEL
    {
        while (!optimization_end) {
            OMP_SINGLE
            {
                start_time_ = timer_.get_time();
                if(accel && reset)
                {
                    AA_reset_time_start = timer_.get_time();
                }
                prev_Y_ = Y;
            }

            update_G_GY();
            update_Y_GY();
            update_dual_variables_GY();

            OMP_SINGLE
            {
                iter_num++;
                optimization_converge = need_compute_residual_norms
                        && (primal_residual_sqr_norm <= primal_residual_sqr_norm_threshold
                            && dual_residual_sqr_norm <= dual_residual_sqr_norm_threshold);
                optimization_end = optimization_converge || iter_num >= param.grad_solver_max_iter ;
                output_progress = need_compute_residual_norms && (iter_num % param.grad_solver_output_frequency == 0);

                if (optimization_converge) {
                    std::cout << "Solver converged." << std::endl;
                } else if (optimization_end) {
                    std::cout << "Maximum number of iterations reached." << std::endl;
                }

                if (output_progress || optimization_end) {
                    std::cout << "Iteration " << iter_num << ":" << std::endl;
                    std::cout << "Primal residual squared norm: " << primal_residual_sqr_norm
                              << ",  threshold:" << primal_residual_sqr_norm_threshold
                              << std::endl;
                    std::cout << "Dual residual squared norm: " << dual_residual_sqr_norm
                              << ",  threshold:" << dual_residual_sqr_norm_threshold
                              << std::endl;

                    if (optimization_end)
                        std::cerr << iter_num << '\t';
                }

                //            // Modified features by
                //#ifdef CERR_RESIDUALS
                //            //        std::cerr << iter_num << "\t" << primal_residual_sqr_norm << "\t" << dual_residual_sqr_norm << std::endl;
                //#endif

                if (!optimization_end)
                {
                    Timer::EventID iter_time_ = timer_.get_time();
                    total_time += timer_.elapsed_time(start_time_, iter_time_);

                    if(need_compute_residual_norms)
                    {
                        current_residual = primal_residual_sqr_norm + dual_residual_sqr_norm;
                    } else {
                        current_residual = (new_Y_ - (*current_SG)).colwise().squaredNorm().dot(Y_area)
                                + (new_Y_ - prev_Y_).colwise().squaredNorm().dot(Y_area_squared);
                    }
                    //std::cout << "current_residual = " << current_residual << std::endl;

                    if(accel && reset)
                    {
                        AA_reset_time += timer_.elapsed_time(AA_reset_time_start, timer_.get_time());
                    }

                    accept = (!accel) || reset || current_residual < prev_residual;

                    if (accept) {
                        if (reset) {
                            //std::cout << ", AA was reset";
                            count_reject++;
                        }
                        //std::cout << std::endl;

                        prev_residual = current_residual;
                        reset = false;

                        function_values_.push_back(current_residual);
                        elapsed_time_.push_back(total_time + AA_compute_time);

                        Timer::EventID AA_time = timer_.get_time();
                        if (accel) {
                            default_Y_ = new_Y_;
                            default_D_ = new_D_;

                            aa->compute(new_Y_, new_D_, Y, D);

                        } else {
                            Y = new_Y_;
                            D = new_D_;
                        }
                        AA_compute_time += timer_.elapsed_time(AA_time, timer_.get_time());
                    } else {
                        Timer::EventID AA_time = timer_.get_time();
                        Y = default_Y_;
                        D = default_D_;
                        reset = true;
                        iter_num--;

                        //if (accel) {
                        aa->reset(Y, D);
                        AA_compute_time += timer_.elapsed_time(AA_time, timer_.get_time());
                        //std::cout << "iter = " << iter_num << ", reject." << std::endl;
                        //}
                    }
                }

            }

            //        OMP_SINGLE
            //        {
            //            std::swap(current_SG, prev_SG);
            //        }
        }
    }

    std::cout << "reject number = " << count_reject << std::endl;
    double solving_time = timer_.elapsed_time(start_time_, timer_.get_time());
    std::cout << "Solving time: " << solving_time << " secs." << std::endl;
    std::cout << "AA reset time: " << AA_reset_time << " secs." << std::endl;
    std::cout << "AA compute time: " << AA_compute_time << " secs." << std::endl;

    save(accel, param.Anderson_m, param.acceleration_type);

    if (aa) {
        delete aa;
    }
}

void GeodesicDistanceSolver::compute_integrable_gradients_DR() {
    optimization_end = false;
    iter_num = 0;
    bool accel = param.Anderson_m > 0;
    bool accept = false, reset = false;
    double prev_prim_residual = std::numeric_limits<double>::max();
    double current_residual = 0;
    U = D;
    V = D;

    std::cout << "current_SG dim = " << current_SG->size() << std::endl;

    OMP_PARALLEL
    {
        OMP_FOR
                for (int i = 0; i < n_interior_edges; i++)
        {
            D.col(2 * i) = Y.col(2 * i);// *sqrt( Y_area(2 * i));
            D.col(2 * i + 1) = Y.col(2 * i + 1);// * sqrt(Y_area(2 * i + 1));
        }
        update_Y_DR();
        update_U_DR();
        update_G_DR();
        update_V_DR();
    }
    D = D + V - U;
    new_Y_ = Y;
    new_U_ = U;

    aa = new AndersonAcceleration(param.Anderson_m,
                                  D.size(),
                                  D.size());
    aa->init(D);

    Timer timer_;
    Timer::EventID start_time_;
    int count_reject = 0;
    double AA_reset_time = 0.0, AA_compute_time = 0.0, total_time = 0.0;
    Timer::EventID AA_reset_time_start;

    while (!optimization_end) {
        start_time_ = timer_.get_time();
        if(reset)
        {
            AA_reset_time_start = timer_.get_time();
        }
        prev_U_ = U;

        OMP_PARALLEL
        {
            update_Y_DR();
            update_U_DR();
            update_G_DR();
            update_V_DR();
        }
        update_dual_variables_DR();

        iter_num++;

        optimization_end = iter_num >= param.grad_solver_max_iter ;

        if (!optimization_end)
        {
            primal_residual_sqr_norm = (V - U).colwise().squaredNorm().dot(Y_area);
            //std::cout << " primal_residual_sqr_norm= " << primal_residual_sqr_norm;

            if(reset)
            {
                AA_reset_time += timer_.elapsed_time(AA_reset_time_start, timer_.get_time());
            }

            accept = reset || primal_residual_sqr_norm < prev_prim_residual;

            if (accept) {
                if (reset) {
                    //std::cout << ", AA was reset" << std::endl;
                    count_reject++;
                }

                prev_prim_residual = primal_residual_sqr_norm;
                reset = false;

                Timer::EventID iter_time_ = timer_.get_time();
                total_time += timer_.elapsed_time(start_time_, iter_time_);
                elapsed_time_.push_back(total_time + AA_compute_time);

                //Calculate the combined residual for comparison, not need in application.
                //So we don't count this part of time in.
                OMP_PARALLEL
                {

                    OMP_FOR for (int i = 0; i < n_interior_edges; i++)
                    {
                        Matrix32 y = new_D_.block(0, 2 * i, 3, 2);
                        Eigen::Vector3d d = e.col(i) * (e.col(i).dot(y.col(1) - y.col(0)));
                        double a = Y_area(2 * i) / (Y_area(2 * i) + Y_area(2 * i + 1));
                        y.col(0) += (1.0 - a) * d;
                        y.col(1) -= a * d;
                        new_Y_.block(0, 2 * i, 3, 2) = y;
                    }
                    OMP_FOR
                            for (int i = 0; i < n_interior_edges; i++)
                    {
                        new_U_.col(2 * i) = new_Y_.col(2 * i);// *sqrt( Y_area(2 * i));
                        new_U_.col(2 * i + 1) = new_Y_.col(2 * i + 1);// * sqrt(Y_area(2 * i + 1));
                    }
                }
                dual_residual_sqr_norm = (new_U_ - U).colwise().squaredNorm().dot(Y_area_squared);
                primal_residual_sqr_norm  = (V - new_U_).colwise().squaredNorm().dot(Y_area);

                current_residual = primal_residual_sqr_norm + dual_residual_sqr_norm;
                //std::cout << " current_residual = " << current_residual << std::endl;
                function_values_.push_back(current_residual);


                Timer::EventID AA_time = timer_.get_time();
                default_D_ = new_D_;
                aa->compute(new_D_, D);
                AA_compute_time += timer_.elapsed_time(AA_time, timer_.get_time());
            } else {
                Timer::EventID AA_time = timer_.get_time();
                D = default_D_;
                reset = true;
                iter_num--;

                aa->replace(D);
                AA_compute_time += timer_.elapsed_time(AA_time, timer_.get_time());
            }
        }
    }

    std::cout << "reject number = " << count_reject << std::endl;
    double solving_time = timer_.elapsed_time(start_time_, timer_.get_time());
    std::cout << "Solving time: " << solving_time << " secs." << std::endl;
    std::cout << "AA reset time: " << AA_reset_time << " secs." << std::endl;
    std::cout << "AA compute time: " << AA_compute_time << " secs." << std::endl;

    save(accel, param.Anderson_m, param.acceleration_type);

    if (aa) {
        delete aa;
    }
}

void GeodesicDistanceSolver::prepare_integrate_geodesic_distance() {
    transition_from_vtx.setConstant(n_vertices, -1);
    transition_edge_vector.setZero(3, n_vertices);
    transition_edge_neighbor_faces.setConstant(2, n_vertices, -1);

    OMP_PARALLEL
    {
        // Set up incident relation between internal edges and faces
        OMP_SINGLE
        {
            Matrix2Xi internal_edge_faces;  // Associated indices for each internal edge
            internal_edge_faces.setConstant(2, n_edges, -1);
            Matrix3X internal_edge_unit_vectors; // Edge vector for each internal edge, corresponding to its first halfedge
            internal_edge_unit_vectors.setZero(3, n_edges);
            n_interior_edges = 0;

            faces_Y_index.setConstant(3, n_faces, -1);  // Indices of internal edges (within the internal edge array) associated with each face
            IndexVector num_rows; // Number of internal edges for each face
            num_rows.setZero(n_faces);

            for (int i = 0; i < n_edges; ++i) {
                MeshType::Edge eh(i);
                if (!mesh.is_boundary(eh)) {
                    for (int k = 0; k < 2; ++k) {
                        int f = mesh.face(mesh.halfedge(eh, k)).idx();
                        internal_edge_faces(k, n_interior_edges) = f;
                        faces_Y_index(num_rows(f)++, f) = 2 * n_interior_edges + k;
                    }

                    internal_edge_unit_vectors.col(n_interior_edges) = edge_vector.col(i).normalized();
                    n_interior_edges++;
                }
            }

            edge_vector.resize(3, 0); // Release edge_vector matrix
            S = internal_edge_faces.block(0, 0, 2, n_interior_edges);
            e = internal_edge_unit_vectors.block(0, 0, 3, n_interior_edges);
        }

        // Pre-computation for integrating gradients
        OMP_FOR
                for (int i = 0; i < n_vertices; ++i) {
            MeshType::Halfedge heh(transition_halfedge_idx(i));
            if (heh.is_valid()) {
                MeshType::Vertex from_vh = mesh.from_vertex(heh);
                Eigen::Vector3d edge_vec = to_eigen_vec3d(
                            mesh.position(mesh.to_vertex(heh)) - mesh.position(from_vh));
                Eigen::Vector2i face_idx;
                face_idx(0) = mesh.face(heh).idx();
                face_idx(1) = mesh.face(mesh.opposite_halfedge(heh)).idx();
                transition_from_vtx(i) = from_vh.idx();
                transition_edge_vector.col(i) = edge_vec;
                transition_edge_neighbor_faces.col(i) = face_idx;
            }
        }

        OMP_SINGLE
        {
            mesh.clear();
            transition_halfedge_idx.resize(0);
            D.setZero(3, 2 * n_interior_edges);
            G = init_grad;
            Y.setZero(3, 2 * n_interior_edges);
            Y_area.resize(2 * n_interior_edges);
            SG1.setZero(3, 2 * n_interior_edges);
            SG2.setZero(3, 2 * n_interior_edges);
            current_SG = &SG1;
            prev_SG = &SG2;
        }

        OMP_FOR
                for (int i = 0; i < n_interior_edges; i++) {
            current_SG->col(2 * i) = G.col(S(0, i));
            current_SG->col(2 * i + 1) = G.col(S(1, i));
        }

        OMP_FOR
                for (int i = 0; i < n_interior_edges; i++) {
            Y_area(2 * i) = face_area(S(0, i));
            Y_area(2 * i + 1) = face_area(S(1, i));
        }

        OMP_SINGLE
        {
            Y_area /= Y_area.mean();
            Y_area_squared = Y_area.array().square().matrix();
            primal_residual_sqr_norm_threshold = Y_area.sum() * param.grad_solver_eps * param.grad_solver_eps;
            dual_residual_sqr_norm_threshold = Y_area_squared.sum() * param.grad_solver_eps * param.grad_solver_eps;
            (*prev_SG) = (*current_SG);
        }
    }
}

void GeodesicDistanceSolver::integrate_geodesic_distance() {
    geod_dist_values.setZero(n_vertices);
    int n_segments = bfs_segment_addr.size() - 1;
    int segment_count = 1;  // We update the distance values starting from the second layer of BFS vertex list
    int segment_begin_addr, segment_end_addr;
    bool end_propagation = false;

    OMP_PARALLEL
    {
        while (!end_propagation) {
            OMP_SINGLE
            {
                segment_begin_addr = bfs_segment_addr(segment_count);
                segment_end_addr = bfs_segment_addr(segment_count + 1);
            }

            OMP_FOR
                    for (int i = segment_begin_addr; i < segment_end_addr; ++i) {
                double from_d = geod_dist_values(transition_from_vtx(i));
                Eigen::Vector3d grad = Eigen::Vector3d::Zero();
                Eigen::Vector2i neighbor_faces = transition_edge_neighbor_faces.col(i);
                int n_neighbor_faces = 0;
                for (int k = 0; k < 2; ++k) {
                    if (neighbor_faces(k) >= 0) {
                        grad += G.col(neighbor_faces(k));
                        n_neighbor_faces++;
                    }
                }

                grad /= double(n_neighbor_faces);
                geod_dist_values(bfs_vertex_list(i)) = from_d + transition_edge_vector.col(i).dot(grad);
            }

            OMP_SINGLE
            {
                segment_count++;
                end_propagation = segment_count >= n_segments;
            }
        }
    }

    // Recover geodesic distance in the original scale
    geod_dist_values *= model_scaling_factor;
}

void GeodesicDistanceSolver::update_Y() {
    OMP_FOR
            for (int i = 0; i < n_interior_edges; i++)
    {
        Matrix32 y = prev_SG->block(0, 2 * i, 3, 2) - D.block(0, 2 * i, 3, 2);
        Eigen::Vector3d d = e.col(i) * (e.col(i).dot(y.col(1) - y.col(0)));
        double a = Y_area(2 * i) / (Y_area(2 * i) + Y_area(2 * i + 1));
        y.col(0) += (1.0 - a) * d;
        y.col(1) -= a * d;
        Y.block(0, 2 * i, 3, 2) = y;
    }
}

void GeodesicDistanceSolver::update_G() {
    OMP_FOR
            for (int i = 0; i < n_faces; i++)
    {
        int n_aux_var = 0;
        Eigen::Vector3d R = Eigen::Vector3d::Zero();
        for (int j = 0; j < 3; j++) {
            int index = faces_Y_index(j, i);
            if (index >= 0) {
                R += (Y.col(index) + D.col(index));
                n_aux_var++;
            }
        }

        double w = 2.0 / param.penalty;
        R += w * init_grad.col(i);
        R /= (w + n_aux_var);
        new_G_.col(i) = R;
    }
}

void GeodesicDistanceSolver::update_dual_variables()
{
    OMP_FOR
            for (int i = 0; i < n_interior_edges; ++ i) {
        current_SG->col(2 * i) = new_G_.col(S(0, i));
        current_SG->col(2 * i + 1) = new_G_.col(S(1, i));
    }

    OMP_SINGLE
    {
        need_compute_residual_norms
                = ((iter_num+1) % param.grad_solver_convergence_check_frequency == 0);
    }

    OMP_SECTIONS
    {
        OMP_SECTION
        {
            if(need_compute_residual_norms)
            {
                primal_residual_sqr_norm =
                        (Y - (*current_SG)).colwise().squaredNorm().dot(Y_area);
            }
        }

        OMP_SECTION
        {
            if(need_compute_residual_norms)
            {
                dual_residual_sqr_norm = ((*current_SG) - (*prev_SG)).colwise().squaredNorm().dot(Y_area_squared);
            }
        }

        OMP_SECTION
        {
            new_D_ = D + Y - (*current_SG);
        }

    }

}

void GeodesicDistanceSolver::update_G_GY() {
    OMP_FOR
            for (int i = 0; i < n_faces; i++)
    {
        int n_aux_var = 0;
        Eigen::Vector3d R = Eigen::Vector3d::Zero();
        for (int j = 0; j < 3; j++) {
            int index = faces_Y_index(j, i);
            if (index >= 0) {
                R += (Y.col(index) + D.col(index));
                n_aux_var++;
            }
        }

        double w = 2.0 / param.penalty;
        R += w * init_grad.col(i);
        R /= (w + n_aux_var);
        G.col(i) = R;
    }
}

void GeodesicDistanceSolver::update_Y_GY() {
    OMP_FOR
            for (int i = 0; i < n_interior_edges; ++ i) {
        current_SG->col(2 * i) = G.col(S(0, i));
        current_SG->col(2 * i + 1) = G.col(S(1, i));
    }

    OMP_FOR
            for (int i = 0; i < n_interior_edges; i++)
    {
        Matrix32 y = current_SG->block(0, 2 * i, 3, 2) - D.block(0, 2 * i, 3, 2);
        Eigen::Vector3d d = e.col(i) * (e.col(i).dot(y.col(1) - y.col(0)));
        double a = Y_area(2 * i) / (Y_area(2 * i) + Y_area(2 * i + 1));
        y.col(0) += (1.0 - a) * d;
        y.col(1) -= a * d;
        new_Y_.block(0, 2 * i, 3, 2) = y;
    }
}

void GeodesicDistanceSolver::update_dual_variables_GY()
{
    OMP_SINGLE
    {
        need_compute_residual_norms
                = ((iter_num+1) % param.grad_solver_convergence_check_frequency == 0);
    }

    OMP_SECTIONS
    {
        OMP_SECTION
        {
            if(need_compute_residual_norms)
            {
                primal_residual_sqr_norm =
                        (new_Y_ - (*current_SG)).colwise().squaredNorm().dot(Y_area);
            }
        }

        OMP_SECTION
        {
            if(need_compute_residual_norms)
            {
                dual_residual_sqr_norm = (new_Y_ - prev_Y_).colwise().squaredNorm().dot(Y_area_squared);
            }
        }

        OMP_SECTION
        {
            new_D_ = D + new_Y_ - (*current_SG);
        }

    }

}

void GeodesicDistanceSolver::update_Y_DR() {

    OMP_FOR
            for (int i = 0; i < n_interior_edges; i++)
    {
        Matrix32 y = D.block(0, 2 * i, 3, 2);
//        y.col(1) /= sqrt(Y_area(2 * i + 1));
//        y.col(0) /= sqrt(Y_area(2 * i));
        Eigen::Vector3d d = e.col(i) * (e.col(i).dot(y.col(1) - y.col(0)));
        double a = Y_area(2 * i) / (Y_area(2 * i) + Y_area(2 * i + 1));
        y.col(0) += (1.0 - a) * d;
        y.col(1) -= a * d;
        Y.block(0, 2 * i, 3, 2) = y;
    }
}

void GeodesicDistanceSolver::update_U_DR() {

    OMP_FOR
            for (int i = 0; i < n_interior_edges; i++)
    {
        U.col(2 * i) = Y.col(2 * i);// *sqrt( Y_area(2 * i));
        U.col(2 * i + 1) = Y.col(2 * i + 1);// * sqrt(Y_area(2 * i + 1));
    }
}

void GeodesicDistanceSolver::update_G_DR() {
    OMP_FOR
            for (int i = 0; i < n_faces; i++)
    {
        int n_aux_var = 0;
        Eigen::Vector3d R = Eigen::Vector3d::Zero();
        for (int j = 0; j < 3; j++) {
            int index = faces_Y_index(j, i);
            if (index >= 0) {
                R += (2.0*U.col(index) - D.col(index));
                n_aux_var++;
            }
        }

        double w = 2.0 / param.penalty;
        R += w * init_grad.col(i);
        R /= (w + n_aux_var);
        G.col(i) = R;
    }
}

void GeodesicDistanceSolver::update_V_DR() {

    OMP_FOR
            for (int i = 0; i < n_interior_edges; ++ i) {
        V.col(2 * i) = G.col(S(0, i));// * sqrt(Y_area(2 * i));
        V.col(2 * i + 1) = G.col(S(1, i));// * sqrt(Y_area(2 * i + 1));
    }
}

void GeodesicDistanceSolver::update_dual_variables_DR()
{
    new_D_ = D + V - U;
}

void GeodesicDistanceSolver::save(bool accel, int Anderson_m, int acceleration_type)
{
    std::string file;
    if (accel){
        if(acceleration_type == 2)
            file = "./result/residual-DR-"+std::to_string(Anderson_m)+".txt";
        else
            file = "./result/residual-AA-"+std::to_string(Anderson_m)+".txt";
    }
    else
        file = "./result/residual-no.txt";

    std::ofstream ofs;
    ofs.open(file, std::ios::out | std::ios::ate);
    if (!ofs.is_open())
    {
        std::cout << "Cannot open: " << file << std::endl;
    }

    ofs << std::setprecision(16);
    for (size_t i = 0; i < elapsed_time_.size(); i++)
    {
        ofs << elapsed_time_[i] << '\t' << sqrt(param.penalty*function_values_[i]/current_SG->size())  << std::endl;
    }

    ofs.close();

    clear_iteration_history();
}
