
#include "MCL/RenderWindow.hpp"

class SliceController : public mcl::Controller {
public:
	bool process_slice;
	float slice_fraction;
	float slice_step;
	bool save_single_ss;
	SliceController() : process_slice(true), slice_fraction(0.5f), slice_step(0.07f), save_single_ss(false) {}

	void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){
		mcl::Controller::key_callback(window,key,scancode,action,mods);
		if( action != GLFW_PRESS ){ return; }
		switch ( key ){
			case GLFW_KEY_UP:{
				slice_fraction = std::min( 0.9f, slice_fraction + slice_step );
				process_slice = true;
			} break;
			case GLFW_KEY_DOWN:{
				slice_fraction = std::max( -0.1f, slice_fraction - slice_step );
				process_slice = true;
			} break;
			case GLFW_KEY_LEFT:{
				slice_step = std::max( 0.01f, slice_step - 0.01f );
				std::cout << "New slice step: " << slice_step << std::endl;
			} break;
			case GLFW_KEY_RIGHT:{
				slice_step = std::min( 0.3f, slice_step + 0.01f );
				std::cout << "New slice step: " << slice_step << std::endl;
			} break;
			case GLFW_KEY_S:{
				save_single_ss = true;
			} break;
			default: break;
		}
	}
};


class SliceViewer {
public:
	SliceViewer(){
		m_rw = std::make_shared<mcl::RenderWindow>( mcl::RenderWindow() );
		m_c = std::make_shared<SliceController>( SliceController() );
		m_rw->set_controller( m_c );
	}

	// Sets the tet mesh to be sliced (viewed)
	inline void set_mesh( std::shared_ptr<mcl::TetMesh> mesh_ );

	// Slices a mesh as a fraction of the AABB
	// Input is slice_fraction from SliceController.
	// Sets slicedMesh and renderMesh.
	inline void slice_mesh();

	// Returns success or failure
	inline bool display();

private:
	std::shared_ptr<mcl::RenderWindow> m_rw;
	std::shared_ptr<SliceController> m_c;
	std::shared_ptr<mcl::TetMesh> initMesh;
	std::shared_ptr<mcl::TetMesh> slicedMesh;
	std::shared_ptr<mcl::RenderMesh> renderMeshSolid;
	std::shared_ptr<mcl::RenderMesh> renderMeshWire;
	Eigen::AlignedBox<float,3> aabb;
};


inline void SliceViewer::set_mesh( std::shared_ptr<mcl::TetMesh> mesh_ ){

	initMesh = mesh_;
	initMesh->need_faces();
	initMesh->need_normals();
	aabb = initMesh->bounds();

	slicedMesh = mcl::TetMesh::create();
	slicedMesh->vertices = initMesh->vertices;
	slicedMesh->tets = initMesh->tets;
	slicedMesh->need_faces();
	slicedMesh->need_normals();
	renderMeshSolid = mcl::RenderMesh::create( slicedMesh, mcl::RenderMesh::DYNAMIC );
	renderMeshWire = mcl::RenderMesh::create( slicedMesh, mcl::RenderMesh::DYNAMIC | mcl::RenderMesh::WIREFRAME );

	renderMeshWire->phong.diff.setZero();
	renderMeshWire->phong.amb.setZero();
	renderMeshWire->phong.spec.setZero();

} // end set mesh


inline void SliceViewer::slice_mesh(){

	const std::vector<mcl::Vec3f> &verts = initMesh->vertices;

	// Compute cutoff
	int axis = 2; // z?
	mcl::Vec3f diag = aabb.max() - aabb.min();
	mcl::Vec3f cutoff = aabb.min() + (diag * m_c->slice_fraction);

	slicedMesh->tets.clear();
	int n_tets = initMesh->tets.size();
	for( int i=0; i<n_tets; ++i ){
		mcl::Vec4i &tet = initMesh->tets[i];
		mcl::Vec3f center(0,0,0);
		for( int j=0; j<4; ++j ){ center += verts[ tet[j] ]; }
		center *= 0.25f;

		if( center[axis] > cutoff[axis] ){
			slicedMesh->tets.emplace_back( tet );
		}
	}
	
	// Update normals and faces
	slicedMesh->need_faces(true);
	slicedMesh->need_edges(true,true);
	slicedMesh->need_normals(true);
	renderMeshWire->load_buffers();
	renderMeshSolid->load_buffers();

} // end slice mesh


// Returns success or failure
inline bool SliceViewer::display(){

	std::cout << "SliceViewer controls:" <<
		"\n\t UP/DOWN to move the slice" <<
		"\n\t LEFT/RIGHT change slice step size" <<
		"\n\t S to save a screenshot" <<
	std::endl;

	// Create opengl context
	GLFWwindow* window = m_rw->init();
	if( !window ){ return false; }

	// Initialize with all verts/faces to make sure
	// GL buffer data is set with max possible verts/inds.
	renderMeshWire->load_buffers();
	renderMeshSolid->load_buffers();
	m_rw->add_mesh( renderMeshWire );
	m_rw->add_mesh( renderMeshSolid );

	// Set the camera to a sensible location
	m_rw->nice_camera_location();

	// Game loop
	while( m_rw->is_open() ){

		if( m_c->process_slice ){
			m_c->process_slice = false;
			slice_mesh();
		}

		m_rw->draw();
		glfwPollEvents();

		// Save screenshot?
		if( m_c->save_single_ss ){
			m_c->save_single_ss = false;
			std::string fn = "sliceviewer_screenshot.png";
			m_rw->save_screenshot( fn );
			std::cout << "Saving screenshot as " << fn << std::endl;
		}

	} // end game loop

	return true;

} // end display

