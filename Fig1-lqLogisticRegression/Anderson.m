% BSD 3-Clause License
% 
% Copyright (c) 2020, Yuxin Yao
% 
% Redistribution and use in source and binary forms, with or without
% modification, are permitted provided that the following conditions are met:
% 
% * Redistributions of source code must retain the above copyright notice, this
% list of conditions and the following disclaimer.
% 
% * Redistributions in binary form must reproduce the above copyright notice,
% this list of conditions and the following disclaimer in the documentation
% and/or other materials provided with the distribution.
% 
% * Neither the name of the copyright holder nor the names of its
% contributors may be used to endorse or promote products derived from
% this software without specific prior written permission.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
% FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
% DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
% SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
% CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
% OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
% OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

classdef Anderson  < handle
   properties   
       dim;
       effdim;
       mk;
       M;
       cu;
       cf;
       gg;
       df;
       dg;
       coef;
       iter_count=0;
       c_idx=0;
   end  
     
   methods  
      function obj = Anderson(x0,num,edim) 
         if nargin > 0  
            obj.mk = num;
            obj.effdim=edim;
            obj.dim=length(x0);
	        obj.M=zeros(obj.mk,obj.mk);
            obj.cu = x0;
            obj.cf = zeros(obj.effdim,1);
            obj.gg = zeros(obj.dim,1);
            obj.df = zeros(obj.effdim,obj.mk);
            obj.dg = zeros(obj.dim,obj.mk);
            obj.coef = zeros(obj.mk,1);
            obj.iter_count=0;
            obj.c_idx=0;
         end  
      end   
   end  
     
   methods   
       function ss=reset(obj,x)
         obj.cu=x;
         obj.iter_count = 0;
         obj.c_idx = 0;
         ss=x;
       end
      function ss=replace(obj,x)  
         obj.cu=x;
         ss=x;
      end % Modulus get function  
      function result=compute(obj,gx)
	      obj.gg=gx;
          obj.cf=obj.gg(1:obj.effdim)-obj.cu(1:obj.effdim);
        if obj.iter_count==0
            obj.df(:,0+1)=-obj.cf;
            obj.dg(:,0+1)=-obj.gg;
            obj.cu=obj.gg;
        else
            obj.df(:,obj.c_idx+1)=obj.df(:,obj.c_idx+1)+obj.cf;
            obj.dg(:,obj.c_idx+1)=obj.dg(:,obj.c_idx+1)+obj.gg;
            mm=min([obj.mk,obj.iter_count]);
            if mm==1
                obj.coef(0+1)=0;
                fnorm=norm(obj.df(:,0+1));
                obj.M(0+1,0+1)=fnorm^2;
                obj.coef(0+1)=dot(obj.df(:,obj.c_idx+1),obj.cf)/fnorm/fnorm;
            else
                new_inner=obj.df(:,1:mm)'*obj.df(:,obj.c_idx+1);
                obj.M(obj.c_idx+1,1:mm)=new_inner;
                obj.M(1:mm,obj.c_idx+1)=new_inner';
                bb=obj.df(:,1:mm)'*obj.cf;
                obj.coef(1:mm)=lsqminnorm(obj.M(1:mm,1:mm),bb);
            end
            obj.cu=obj.gg-(obj.dg(:,1:mm)*obj.coef(1:mm));
            obj.c_idx=mod(obj.c_idx+1,obj.mk);
            obj.df(:,obj.c_idx+1)=-obj.cf;
            obj.dg(:,obj.c_idx+1)=-obj.gg;
        end
        obj.iter_count=obj.iter_count+1;
        result=obj.cu;
      end
   end % classdef
end