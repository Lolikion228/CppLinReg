#include "LinReg.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <tuple>
#include <limits>
#include <cstring>
#include <random>
#include <chrono>


double dot(double* x1, double* x2, int dim){
    double res = 0;
    for(int i=0; i<dim; ++i){
        res += x1[i] * x2[i];
    }
    return res;
}


LinReg::LinReg(int d, double a, double b){
    dim = d;
    weights = (double*) calloc(dim + 1, sizeof(double));// init with zeros
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(a,b);
    for(int i = 0; i <= dim; ++i) {
        weights[i] = dist(gen);
    }
}


// x[dim]
double LinReg::pred_single(double* x) const{
    double res = dot(x, weights, dim);
    res += weights[dim]; // adding w_0
    return res;
}


// X[n_obj][dim]
double* LinReg::pred_batch(double** X, int n_obj) const{
    double* res = (double*) calloc(n_obj, sizeof(double));
    for(int i=0; i<n_obj; ++i){
        res[i] = pred_single(X[i]);
    }
    return res;
}


// return w from R^(dim+1) where w_0 = w[dim] = bias term
double* LinReg::GetWeights() const{
    double* w = (double*) calloc(dim + 1, sizeof(double));
    memcpy(w, weights, (dim + 1) * sizeof(double));
    return w;
}


// set weights to w where w[dim] = w_0 = bias term
void LinReg::SetWeights(double *w){
    memcpy(weights, w, (dim + 1) * sizeof(double));
    total_epochs = 0;
}


// X[n_obj][dim]
void LinReg::fit(double** X, double* y, int n_obj, LRSchedulerBase& lr, int n_epoch, bool verbose){
    double total_loss = 0;
    double min_loss = std::numeric_limits<double>::max();
    double max_loss = 0;

    auto fit_start = std::chrono::high_resolution_clock::now();

    for(int k=0; k<n_epoch; ++k){
        
        double max_grad = 0;
        double max_w = 0;


        ++total_epochs;
        if(verbose){
            std::cout << "Epoch " << k + 1 << " / " << n_epoch;
        }
        //computing y_pred - y
        double* pred = pred_batch(X, n_obj);
        double* diff = (double*)calloc(n_obj, sizeof(double));

        for(int i=0; i<n_obj; ++i){
            diff[i] = pred[i] - y[i];
        }

        free(pred);



        // computing gradient
        double* grad = (double*)calloc((dim+1), sizeof(double));
        grad[dim] = 0;

        for(int j=0; j<n_obj; ++j){
            grad[dim] += diff[j];
        }

        for(int i=0; i<dim; ++i){
            grad[i] = 0;
            for(int j=0; j<n_obj; ++j){
                grad[i] += X[j][i] * diff[j];
            }
        }

        for(int i=0; i<=dim; ++i){
            grad[i] *= (2.0 / n_obj);
            max_grad = std::max(max_grad, std::abs(grad[i]));
        }



        //applying gradient
        for(int i=0; i<=dim; ++i){
            weights[i] -= lr.Step(k+1) * grad[i];
            max_w = std::max(max_w, std::abs(weights[i]));
        }
        free(grad);

        
        // computing MSE
        double epoch_loss = 0;

        for(int j=0; j<n_obj; ++j){
            epoch_loss += diff[j]*diff[j];
        }

        epoch_loss /= n_obj;
        min_loss = std::min(epoch_loss, min_loss);
        max_loss = std::max(epoch_loss, max_loss);
        free(diff);

        
        if(verbose){
            std::cout << "  |  epoch_loss " << epoch_loss << "  |  max_grad " << max_grad << "  |  max_weight " << max_w << "  |  curr_lr " << lr._curr_lr <<"\n";
        }
        total_loss += epoch_loss;
        
    }
    auto fit_end = std::chrono::high_resolution_clock::now();
    auto fit_duration = std::chrono::duration_cast<std::chrono::microseconds>(fit_end - fit_start);

    std::cout << "\ntotal time elapsed " << fit_duration.count() / 1000000.0 << " seconds\n";
    std::cout << "total_mean_loss " << total_loss / n_epoch << "\n";
    std::cout << "min_loss " << min_loss << "\n";
    std::cout << "max_loss " << max_loss << "\n";
  
}


std::ostream& operator << (std::ostream& out, const LinReg& LR){

    out << "LinearRegression model\n";
    out << "dim = " << LR.dim <<"\n";
    out << "was fitted " << LR.total_epochs <<" times\n";

    return out;
}


LinReg::~LinReg(){
            free(weights);
        }


std::tuple<double**, double*, int, int> process_data(int max_obj){
    // data reading
    std::ifstream file("./data/data.txt");
    int n_obj;
    int dim;
    file >> n_obj;
    file >> dim;
    n_obj = std::min(n_obj, max_obj);
    double** X = (double**)calloc(n_obj, sizeof(double*));

    for(int i=0; i<n_obj; ++i){
        X[i] = (double*)calloc(dim, sizeof(double));
    }

    double* y = (double*)calloc(n_obj, sizeof(double));
    double* max_val = (double*)calloc(dim+1, sizeof(double));

    for(int i=0; i<n_obj; ++i){
        for(int j=0; j<dim; ++j){
            file >> X[i][j];
            max_val[j] = std::max(max_val[j], std::abs(X[i][j]));
        }
        file >> y[i];
        max_val[dim] = std::max(max_val[dim], std::abs(y[i]));
    }

    file.close();



    // data normalizing  (min_max)
    for(int i=0; i<n_obj; ++i){
        for(int j=0; j<dim; ++j){
            X[i][j] /= max_val[j];
        }
        y[i] /= max_val[dim];
    }

    free(max_val);

    std::tuple<double**, double*, int, int> result = {X,y,dim,n_obj};
    return result;
}


void free_data(double** X, double* y, int n_obj){

    for(int i=0; i<n_obj; ++i){
        free(X[i]);
    }
    free(y);
    free(X);
}




LRSchedulerBase::LRSchedulerBase(double initial_lr, double decay_rate ){
    _initial_lr = initial_lr;
    _curr_lr = initial_lr;
    _decay_rate = decay_rate;
};



ConstantLR::ConstantLR(double lr):
    LRSchedulerBase(lr, 1.0){}

 double ConstantLR::Step(int epoch){
    return _initial_lr;
 }   

 std::ostream& operator << (std::ostream& out, const ConstantLR& lr){
    std::cout << "ConstantLR scheduler\n";
    std::cout << "lr(epoch) = initial_lr\n";
    std::cout << "lr = " << lr._initial_lr << "\n";
    return out;
 }



StepDecay::StepDecay(double initial_lr, double decay_rate, int step_size ):
    LRSchedulerBase(initial_lr, decay_rate){
        _step_size = step_size; 
    }
    
double StepDecay::Step(int epoch) {
    double new_lr =  _initial_lr * pow(_decay_rate, floor(epoch / _step_size));
    _curr_lr = new_lr;
    return _curr_lr;
}

 std::ostream& operator << (std::ostream& out, const StepDecay& lr){
    std::cout << "StepDecay scheduler\n";
    std::cout << "lr(epoch) = initial_lr * pow(decay_rate, floor(epoch / step_size))\n";
    std::cout << "initial_lr = " << lr._initial_lr << "\n";
    std::cout << "current_lr = " << lr._curr_lr << "\n";
    std::cout << "decay_rate = " << lr._decay_rate << "\n";
    std::cout << "step_size = " << lr._step_size << "\n";
    return out;
 }



ExponentialDecay::ExponentialDecay(double initial_lr, double decay_rate):
    LRSchedulerBase(initial_lr, decay_rate){}

double ExponentialDecay::Step(int epoch) {
    double new_lr =  _initial_lr * pow(_decay_rate, epoch);
    _curr_lr = new_lr;
    return _curr_lr;
}

std::ostream& operator << (std::ostream& out, const ExponentialDecay& lr){
    std::cout << "ExponentialDecay scheduler\n";
    std::cout << "lr(epoch) = initial_lr * pow(decay_rate, epoch)\n";
    std::cout << "initial_lr = " << lr._initial_lr << "\n";
    std::cout << "current_lr = " << lr._curr_lr << "\n";
    std::cout << "decay_rate = " << lr._decay_rate << "\n";
    return out;
 }



CosineDecay::CosineDecay(double initial_lr, double min_lr, int T):
    LRSchedulerBase(initial_lr, 1.0){
        _min_lr = min_lr;
        _T = T;
    }
    
double CosineDecay::Step(int epoch) {
    double new_lr =  _min_lr + 0.5 * (_initial_lr - _min_lr) * (1.0 + cos((epoch * M_PI) / _T));
    _curr_lr = new_lr;
    return _curr_lr;
}

std::ostream& operator << (std::ostream& out, const CosineDecay& lr){
    std::cout << "CosineDecay scheduler\n";
    std::cout << "lr(epoch) =  min_lr + 0.5 * (initial_lr - min_lr) * (1.0 + cos((epoch * M_PI) / T))\n";
    std::cout << "initial_lr = " << lr._initial_lr << "\n";
    std::cout << "current_lr = " << lr._curr_lr << "\n";
    std::cout << "min_lr = " << lr._min_lr << "\n";
    std::cout << "T = " << lr._T << "\n";
    return out;
 }