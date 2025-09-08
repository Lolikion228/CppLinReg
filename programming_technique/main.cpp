#include <iostream>
#include "LinReg.h"
#include <cmath>




int main(){

    // int n_epochs=256;
    // ConstantLR* lr = new ConstantLR(1.1);

   
    // int n_epochs = 256;
    // double initial_lr = 1.0;
    // double decay_rate = 0.97;
    // ExponentialDecay* lr = new ExponentialDecay(initial_lr, decay_rate);
    
    int n_epochs = 512;
    double initial_lr = 1.0;
    double min_lr = 1e-7;
    int T = 64;
    CosineDecay* lr = new CosineDecay(initial_lr, min_lr, T);
   
    // int n_epochs = 512;
    // double initial_lr = 1.0;
    // double decay_rate = 0.95;
    // int step_size = 16;
    // StepDecay* lr = new StepDecay(initial_lr, decay_rate, step_size);

   
    auto [X, y, dim, n_obj] = process_data(30000);

    LinReg LR1(dim, -0.1, 0.1);

    LR1.fit(X, y, n_obj, *lr, n_epochs, false);

    free_data(X, y, n_obj);

    std::cout << "\n" <<*lr;
    delete lr;

}


