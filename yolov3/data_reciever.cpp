#include <iostream>
#include <string>
#include <queue>
#include <cstring>

#define BUFFERSIZE 64

using namespace std;

int main() {

    char line[64];
    char *savePtr, *p;
    double command[4];
    int i;

    cout << "[INFO] C++ Program Opened and Initialized\n";

    while(1){
        std::cin.getline(line, 64, '#');
        //std::cout << line << endl;
        std::cout.flush();

        p = strtok_r(line, " ", &savePtr);
        i = 0;

        while(p != NULL){
            command[i] = strtod(p, NULL);
            p = strtok_r(NULL, " ", &savePtr);
            //cout << ">" << command[i] << "<" << std::endl;
            i++;
        }

        if (command[0] == 0){
            cout << "Command 0" << std::endl;
        } else if (command[0] == 1){
            cout << "Command 1" << std::endl;
        } else if (command[0] == 2){
            cout << "Command 2" << std::endl;
        } else if (command[0] == 3){
            cout << "Command 3, rotate" << std::endl;
        }
    }

    return(0);
}