#ifndef LINREG_H
#define LINREG_H


#include <iostream>
#include <vector>
#include <fstream>
#include <tuple>
#include <limits>
#include <cstring>
#include <cmath>


double dot(double* x1, double* x2, int dim);

class LRSchedulerBase{
    public:

        double _initial_lr;
        double _decay_rate;
        double _curr_lr;

        LRSchedulerBase(double initial_lr, double decay_rate );

        virtual double Step(int epoch) = 0;

        virtual ~LRSchedulerBase(){}
};


class LinReg{
    private:
        
        int dim;
        double* weights;
        int total_epochs = 0;

    public:

        LinReg(int d, double a, double b);

    
        // x[dim]
        double pred_single(double* x) const;


        // X[n_obj][dim]
        double* pred_batch(double** X, int n_obj) const;


        // return w from R^(dim+1) where w_0 = w[dim] = bias term
        double* GetWeights() const;


        // set weights to w where w[dim] = w_0 = bias term
        void SetWeights(double *w);


        // X[n_obj][dim]
        void fit(double** X, double* y, int n_obj, LRSchedulerBase& lr, int n_epoch, bool verbose);


        friend std::ostream& operator << (std::ostream& out, const LinReg& LR);


        ~LinReg();

};

/*
data filet format should be like:

n_obj dim
x[0][0] x[0][1]  ...  x[0][dim - 1]  y[0]
...
x[n_obj - 1][0]  ...  x[n_obj - 1][dim - 1]  y[n_obj - 1]

*/
std::tuple<double**, double*, int, int> process_data(const std::string& data_path, int max_obj);

std::tuple<double**, double*, int, int> process_data(const std::string& data_path);

void free_data(double** X, double* y, int n_obj);






class StepDecay : public LRSchedulerBase{
    private:
        int _step_size;
    public:
        StepDecay(double initial_lr, double decay_rate, int step_size );
            
        double Step(int epoch) override;

        friend std::ostream& operator << (std::ostream& out, const StepDecay& lr);

};



class ConstantLR : public LRSchedulerBase{
    public:
        ConstantLR(double lr);
            
        double Step(int epoch) override;

        friend std::ostream& operator << (std::ostream& out, const ConstantLR& lr);

};




class ExponentialDecay : public LRSchedulerBase{
    public:
        ExponentialDecay(double initial_lr, double decay_rate);
        
        double Step(int epoch) override;

        friend std::ostream& operator << (std::ostream& out, const ExponentialDecay& lr);
};


class CosineDecay : public LRSchedulerBase{
    private:
        double _min_lr;
        int _T;
    public:
        CosineDecay(double initial_lr, double min_lr, int T);
            
        double Step(int epoch) override;

        friend std::ostream& operator << (std::ostream& out, const CosineDecay& lr);
};

#endif // LINREG_H