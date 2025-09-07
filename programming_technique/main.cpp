#include <iostream>
#include <vector>
#include <string>
#include <cstring>

template<typename T, typename Func>
void map(T& data, Func function){
    for(auto& x: data){
        x = function(x);
    }  
}


class String{
    private:
        
        
    public:
    char* data;
        int len;
        
        String(std::string s){
            std::cout << "normal constructor\n";
            len = s.size();
            data = (char*)calloc(len, sizeof(char));
            for(size_t i=0; i<len; ++i){
                data[i] = s[i];
            }
        }


        String(const String& other){
            std::cout << "copy constructor\n";
            len = other.len;
            data = (char*)calloc(len, sizeof(char));
            memcpy(data, other.data, len * sizeof(char));
    
        }


        String(const String&& other){
            std::cout << "move constructor\n";
            len = other.len;
            data = (char*)calloc(len, sizeof(char));
            memcpy(data, other.data, len * sizeof(char));
        }
        
        // оператор присваивания
        String& operator = (const String& other){
            len = other.len;
            free(data);
            data = (char*)calloc(len, sizeof(char));
            memcpy(data, other.data, len * sizeof(char));
            return *this;
        }


        friend std::ostream& operator << (std::ostream &out, const String& str){
            for(int i=0; i<str.len; ++i){
                out << str.data[i];
            }
            return out;
        }


        ~String(){
            std::cout << "destructor\n";
            free(data);
        }


};



String mapping(String& x){
    if (x.data[0] == 'a'){
        return String("A");
    }
    else if(x.data[0] == 'b'){
        return String("B");
    }
    else{
        return String("ERR");
    }
}

int main(){


   std::vector<String> v;

   v.emplace_back("bebra");
   v.emplace_back("amogus");
   v.emplace_back("mogus");

   map(v, mapping);

   std::cout<<v[0] << "\n";
   std::cout<<v[1] << "\n";
   std::cout<<v[2] << "\n";


}