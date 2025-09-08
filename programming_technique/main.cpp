#include <iostream>
#include "LinReg.h"
#include <cmath>




int main(){

    int n_epochs = 64;
    double initial_lr = 1.0;
    double decay = 0.95;
    
    StepDecay lr(n_epochs, initial_lr, decay, 5);
    
    for(int i=1; i<=n_epochs; ++i){
        std::cout << "epoch " << i << "  |  lr " << lr.Step(i) << "\n";

    }

    // auto [X, y, dim, n_obj] = process_data(30000);

    // LinReg LR1(dim, -0.1, 0.1);
    // std::cout << LR1;
    // LR1.fit(X, y, n_obj, 1.0, 0.95, 1e-8, 512, true);
    // std::cout << LR1;

    // free_data(X, y, n_obj);

}


