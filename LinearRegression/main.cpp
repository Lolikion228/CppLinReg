#include <iostream>
#include "LinReg.h"
#include <cmath>


int main(int argc, char**args){

    if ( (argc != 1) && (argc != 5) ){
        throw std::invalid_argument("you should pass 0 or 4 arguments, you passed " + std::to_string(argc-1) + "." );
        return 1;
    }
    
    int n_epochs;
    double initial_lr;
    double min_lr;
    int T;

    if(argc==1){
        n_epochs = 512;
        initial_lr = 1.0;
        min_lr = 1e-7;
        T = 64;
    }
    else if(argc == 5) {
        n_epochs = std::stoi(args[1]);
        initial_lr = std::stod(args[2]);
        min_lr = std::stod(args[3]);
        T = std::stoi(args[4]);
    }

    CosineDecay* lr = new CosineDecay(initial_lr, min_lr, T);

    auto [X, y, dim, n_obj] = process_data("./data/data.txt", 30000);

    LinReg LR1{dim, -0.1, 0.1};

    LR1.fit(X, y, n_obj, *lr, n_epochs, false);
    
    free_data(X, y, n_obj);

    delete lr;
}


