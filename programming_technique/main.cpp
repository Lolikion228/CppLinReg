#include <iostream>
#include <vector>
#include <fstream>
#include <tuple>
#include <limits>
#include <cstring>
#include "LinReg.h"


int main(){


    auto [X, y, dim, n_obj] = process_data(512);

    LinReg LR1(dim);

    std::cout << LR1;
    LR1.fit(X, y, n_obj, 1.0, 128, true);
    std::cout << LR1;
    

    free_data(X, y, n_obj);



}


