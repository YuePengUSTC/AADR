
//
// Helper class for managing OpenGL textures.
//

#ifndef MCL_TEXTURE_H
#define MCL_TEXTURE_H 1

#include <string>
#include <stdio.h>
#include <iostream>
#include "OpenGL.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

namespace mcl {

class Texture {
public:
	Texture() : width(0), height(0), smooth(true), repeated(false), transparency(false), gl_handle(0) {}

	// Destructor frees texture mem
	~Texture(){ if(gl_handle>0){ glDeleteTextures(1, &gl_handle); gl_handle=0; } }

	// Returns whether the texture is valid or not
	inline bool valid(){ return gl_handle!=0; }

	// Returns the OpenGL handle
	inline unsigned int handle() const { return gl_handle; }

	// Creates a texture from file
	inline bool create_from_file( const std::string &filename );
 
	// Creates a texture from memory
	inline bool create_from_memory( int width_, int height_, float *data, bool alpha=false );

	// Turn smoothing off or on
	inline void set_smooth( bool s );

	// Sets repeated on or off
	inline void set_repeated( bool r );

	// Returns size in pixels of the loaded texture
	inline void get_size( int *w, int *h ) const { *w=width; *h=height; }

private:
	int width, height;
	bool smooth, repeated, transparency;
	unsigned int gl_handle;
};


//
//	Implementation
//


inline bool Texture::create_from_file( const std::string &filename ){
/*
	// Load the image with stbi
	int comp;
	unsigned char* image = stbi_load(filename.c_str(), &width, &height, &comp, 0);
	if( image == NULL ){ glDeleteTextures(1, &gl_handle); gl_handle=0; return false; }

	// See if loaded image had transparency
	bool alpha = false;
	if( comp==4 ){ transparency = alpha = true; }

	// Generate texture for use
	if( gl_handle > 0 ){ glDeleteTextures(1, &gl_handle); } // Recreate if needed
	glGenTextures( 1, &gl_handle );
	if( gl_handle == 0 ){
		printf("\n**Texture::create Error: Failed to gen");
		return false;
	}

	// Copy to GPU and set params
	glBindTexture(GL_TEXTURE_2D, gl_handle);
	glTexImage2D(GL_TEXTURE_2D, 0, alpha?GL_RGBA:GL_RGB, width, height, 0, alpha?GL_RGBA:GL_RGB, GL_UNSIGNED_BYTE, image);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, repeated?GL_REPEAT:GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, repeated?GL_REPEAT:GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, smooth?GL_LINEAR:GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, smooth?GL_LINEAR:GL_NEAREST);
	glBindTexture(GL_TEXTURE_2D, 0);

	stbi_image_free(image);
	return true;
*/
	std::cerr << "**TODO: Texture::create_from_file: " << filename << std::endl;
	return false;
}


inline bool Texture::create_from_memory( int width_, int height_, float *data, bool alpha ){

	glGenTextures( 1, &gl_handle );
	if( gl_handle == 0 ){ std::cerr << "\n**Texture::create Error: Failed to gen" << std::endl; return false; }

	width = width_;
	height = height_;
	transparency = alpha;
	glBindTexture( GL_TEXTURE_2D, gl_handle );
	glTexImage2D( GL_TEXTURE_2D, 0, alpha?GL_RGBA:GL_RGB, width, height, 0, alpha?GL_RGBA:GL_RGB, GL_FLOAT, data );
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, repeated?GL_REPEAT:GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, repeated?GL_REPEAT:GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, smooth?GL_LINEAR:GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, smooth?GL_LINEAR:GL_NEAREST);
	glBindTexture( GL_TEXTURE_2D, 0 );
}


inline void Texture::set_smooth( bool s ){
	bool update = ( s != smooth ) && gl_handle != 0;
	smooth = s;
	if( update ){
		glBindTexture( GL_TEXTURE_2D, gl_handle );
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, smooth?GL_LINEAR:GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, smooth?GL_LINEAR:GL_NEAREST);
		glBindTexture( GL_TEXTURE_2D, 0 );
	}
} // end set smooth


inline void Texture::set_repeated( bool r ){
	bool update = ( r != repeated ) && gl_handle != 0;
	repeated = r;
	if( update ){
		glBindTexture( GL_TEXTURE_2D, gl_handle );
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, repeated?GL_REPEAT:GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, repeated?GL_REPEAT:GL_CLAMP_TO_EDGE);
		glBindTexture( GL_TEXTURE_2D, 0 );
	}
} // end set repeated


} // end namespace mcl


#endif
