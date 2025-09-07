#include <iostream>
#include "LinReg.h"



/*
add lr scheduler
add time_counter
*/

int main(){


    auto [X, y, dim, n_obj] = process_data(30000);

    LinReg LR1(dim, -0.1, 0.1);
    // std::cout << LR1;
    LR1.fit(X, y, n_obj, 0.9, 512, true);
    // std::cout << LR1;


    free_data(X, y, n_obj);



}


