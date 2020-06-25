
#ifndef MCL_OPENGL_HPP
#define MCL_OPENGL_HPP

#ifdef MCL_USE_GLEW
#include <GL/glew.h>
#endif
#include <GLFW/glfw3.h> // contains gl includes
#include <sstream>

// Helper function from https://learnopengl.com/#!In-Practice/Debugging
static inline GLenum glCheckError_(const char *file, int line)
{
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        std::stringstream error;
        switch (errorCode)
        {
            case GL_INVALID_ENUM:                  error << "INVALID_ENUM"; break;
            case GL_INVALID_VALUE:                 error << "INVALID_VALUE"; break;
            case GL_INVALID_OPERATION:             error << "INVALID_OPERATION"; break;
//            case GL_STACK_OVERFLOW:                error << "STACK_OVERFLOW"; break;
//            case GL_STACK_UNDERFLOW:               error << "STACK_UNDERFLOW"; break;
            case GL_OUT_OF_MEMORY:                 error << "OUT_OF_MEMORY"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: error << "INVALID_FRAMEBUFFER_OPERATION"; break;
        }
        error << " | " << file << " (" << line << ")\n";
	printf("\n**GL Error: %s",error.str().c_str());
    }
    return errorCode;
}
#define glCheckError() glCheckError_(__FILE__, __LINE__) 

#endif
