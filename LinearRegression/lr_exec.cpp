#include <iostream>
#include "LinReg.h"
#include <cmath>

/*
add:
shuffle
weight penalty
*/


int main(int argc, char**args){

    
    int n_epochs;
    double initial_lr;
    double decay_rate;
    std::string fpath;

    if(argc == 1){
        n_epochs = 512;
        initial_lr = 0.01;
        decay_rate = 0.25;
        fpath = "./data/orig_data.txt";
    }
    else if(argc == 5) {
        n_epochs = std::stoi(args[1]);
        initial_lr = std::stod(args[2]);
        decay_rate = std::stod(args[3]);
        fpath = args[4];
    }
    else{
        throw std::invalid_argument("you should pass 0 or 4 arguments, you passed " + std::to_string(argc-1) + "." );
        return 1;
    }

    ExponentialDecay* lr = new ExponentialDecay(initial_lr, decay_rate);

    auto [X, y, dim, n_obj] = process_data(fpath);

    LinReg LR1{dim, -0.1, 0.1};

    LR1.fit(X, y, n_obj, *lr, n_epochs, false);
    
    free_data(X, y, n_obj);

    delete lr;
}


