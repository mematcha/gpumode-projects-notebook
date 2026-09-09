#include <iostream>
#include <cstdlib>
#include <chrono>
#include <vector>
// cpu_cpp implementation
void cpu_add(const float* x, const float* y, float* z, int n){

    // loop over the vector and add the elements
    for(int i=0;i<n;i++){
        z[i]=x[i]+y[i];
    }

}

int main(int argc, char* argv[]){
    // input vector size from command line
    int n = atoi(argv[1]);

    // allocate the input vectors
    std::vector<float> x(n);
    std::vector<float> y(n);
    std::vector<float> z(n);

    // assign random values to the input vectors
    for(int i=0;i<n;i++){
        x[i]=rand()%100;
        y[i]=rand()%100;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    // call the cpu_add function
    cpu_add(x.data(), y.data(), z.data(), n);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;

    return 0;
}