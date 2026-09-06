#include <iostream> // for input and output operations
#include <vector> // for vector operations
#include <chrono> // for timing operations
#include <numeric> // for numeric operations

// function to add two vectors
// takes in three vectors as input: a, b, and c
// a and b are the input vectors (const references), and c is the output vector (reference not const since it will be modified by addition)
// the function adds the corresponding elements of a and b and stores the result in c
void vector_add_cpu(
    const std::vector<float>& a,
    const std::vector<float>& b, 
    std::vector<float>& c) {
    // loop through the vectors and add the corresponding elements
    for(size_t i=0;i<a.size();i++){
        c[i]=a[i]+b[i];
    }
}
// main function; takes in integer input from command line (argc), and gets a character array also as input (argv)
int main(int argc, char** argv){
    // if the number of arguments is not 2, print an error message and return 1
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <vector_size>" << std::endl;
        return 1;
    }

    // convert the first argument to an integer (argv[0] is actually the name of the program)
    int n = std::stoi(argv[1]);
    // if the integer is less than or equal to 0, print an error message and return 1
    if (n <= 0) {
        std::cerr << "Vector size must be positive." << std::endl;
        return 1;
    }

    // Allocate vectors
    std::vector<float> a(n);
    std::vector<float> b(n);
    std::vector<float> c(n);

    // fill the vectors with the values 0.0, 1.0, 2.0, ..., n-1.0
    //std::iota is a function that fills a container with a sequence of values
    std::iota(a.begin(), a.end(), 0.0);
    std::iota(b.begin(), b.end(), 0.0);

    // start the timer
    auto start_time = std::chrono::high_resolution_clock::now();
    // call the vector_add_cpu function
    vector_add_cpu(a, b, c);
    // stop the timer
    auto end_time = std::chrono::high_resolution_clock::now();
    // calculate the elapsed time
    std::chrono::duration<float> elapsed = end_time - start_time;
    std::cout << "CPU Vector Addition (Size: " << n << ")" << std::endl;
    std::cout << "Total End-to-End Time: " << elapsed.count() << " seconds" << std::endl;

    return 0;
    
}