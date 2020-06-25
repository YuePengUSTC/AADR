
#ifndef MCL_VARMANAGER_H
#define MCL_VARMANAGER_H 1

#include <unordered_map>
//#include <typeinfo>
#include <fstream>
#include <sstream>

namespace mcl {

//
//	Variable Manager class for generic global vars.
//
//	stringstreams are used for casting for i/o.
//	Not very useful for performance-critical applications.
//	Custom types must have stream operators and a default constructor.
//
class VarManager {
public:
	// Checks if a variable exists
	inline bool exists( const std::string &label ) const {
		return string_vars.count(label) > 0;
	}

	// Removes are current variables
	inline void clear() { string_vars.clear(); }

	// Add a variable to the variable manager
	template <typename T>
	inline void set( const std::string &label, T var );

	// Get a variable to the variable manager
	template <typename T>
	inline T get( const std::string &label ) const;

	// If the variable exists, returns true and sets var.
	// Otherwise, returns false and doesn't set var.
	template <typename T>
	inline bool get( const std::string &label, T &var ) const;

	// Saves current variable list to an ascii file. Very simple output:
	// <label1> <value>
	// <label2> <value>
	// ...
	// A variable must have output stream operator for this to work.
	inline void write( const std::string &filename );

	// Reads a config file (in the same format as write).
	// Must have input stream operator to work
	inline void read( const std::string &filename );

	// Returns a string of vars
	inline std::string print() const;

private:

//	class Any { // Custom class to handle all kinds of variables
//	public:
//		Any() : data(nullptr) {}
//		~Any(){ if( data != nullptr ){delete data;} }
//		class Data {
//		public:
//			virtual ~Data() {}
//			virtual std::string toString(){ return ""; }
//			std::string type;
//		};
//		template <typename T> class TypedData : public Data {
//		public:
//			TypedData(T data) : t_data(data) { this->type = typeid(T).name(); }
//			std::string toString(){
//				std::stringstream result;
//				result << t_data;
//				return result.str();
//			}
//			T t_data;
//		};
//		Data *data;
//	}; // end class any

//	std::unordered_map<std::string,Any> vars;
	std::unordered_map<std::string,std::string> string_vars;


}; // end class color map


//
//	Implementation
//


template <typename T>
inline void VarManager::set( const std::string &label, T var ){
	std::stringstream ss;
	ss << var;
	string_vars[label] = ss.str();
//	vars[label] = Any();
//	Any *any = &vars[label];
//	any->data = new Any::TypedData<T>( var );
} // end set


template <typename T>
inline T VarManager::get( const std::string &label ) const {
//	std::unordered_map<std::string,Any>::const_iterator it = vars.find(label);
	std::unordered_map<std::string,std::string>::const_iterator it = string_vars.find(label);
	if( it == string_vars.end() ){
		std::string err = "\n**VarManager Error: " + label + " not found.\n";
		throw std::runtime_error(err.c_str());
	}
//	T value = static_cast<T>( static_cast< Any::TypedData<T>* >(it->second.data)->t_data );
	std::stringstream ss(it->second);
	T value;
	ss >> value;
	return value;
} // end get


template <typename T>
inline bool VarManager::get( const std::string &label, T &var ) const {
//	std::unordered_map<std::string,Any>::const_iterator it = vars.find(label);
	std::unordered_map<std::string,std::string>::const_iterator it = string_vars.find(label);
	if( it == string_vars.end() ){ return false; }
//	var = static_cast<T>( static_cast< Any::TypedData<T>* >(it->second.data)->t_data );
	std::stringstream ss(it->second);
	ss >> var;
	return true;
}


inline void VarManager::write( const std::string &filename ){
	std::ofstream fs;
	fs.open( filename.c_str() );
	std::unordered_map<std::string,std::string>::iterator it = string_vars.begin();
	for( int row = 0; it != string_vars.end(); ++it, ++row ){
		if( row > 0 ){ fs << "\n"; }
		fs << it->first << ' ' << it->second;
	}
	fs.close();
}


inline void VarManager::read( const std::string &filename ){
	std::ifstream infile( filename.c_str() );
	if( infile.is_open() ){
		std::string line;
		while( std::getline( infile, line ) ){
			std::string label, value;
			std::stringstream ss(line);
			ss >> label;
		        getline(ss, value);
			set( label, value );
		}
	} // end file opened
}

inline std::string VarManager::print() const {
	std::stringstream ss;
	std::unordered_map<std::string,std::string>::const_iterator it = string_vars.begin();
	for( int row = 0; it != string_vars.end(); ++it, ++row ){
		if( row > 0 ){ ss << "\n"; }
		ss << it->first << ' ' << it->second;
	}
	return ss.str();
}

} // ns mcl

#endif
