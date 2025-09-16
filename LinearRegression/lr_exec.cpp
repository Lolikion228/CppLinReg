#include <iostream>
#include "LinReg.h"
#include <cmath>


int main(int argc, char**args){

    int n_epochs;
    double initial_lr;
    double decay_rate;
    std::string fpath = "./data/orig_data.txt";
    double regularization_alpha;
    double test_frac = 0.3;

    if(argc == 1){
        n_epochs = 512;
        initial_lr = 0.01;
        decay_rate = 0.25;
        regularization_alpha = 1e-3;
    }
    else if(argc == 5) {
        n_epochs = std::stoi(args[1]);
        initial_lr = std::stod(args[2]);
        decay_rate = std::stod(args[3]);
        regularization_alpha = std::stod(args[4]);
    }
    else{
        throw std::invalid_argument("you should pass 0 or 4 arguments, you passed " + std::to_string(argc-1) + "." );
        return 1;
    }

    ExponentialDecay* lr = new ExponentialDecay(initial_lr, decay_rate);

    auto [X, y, dim, n_obj] = process_data(fpath);

    LinReg LR1{dim, -0.1, 0.1};

    LR1.fit(X, y, test_frac, n_obj, *lr, n_epochs, regularization_alpha, false);
    
    free_data(X, y, n_obj);

    delete lr;
}


