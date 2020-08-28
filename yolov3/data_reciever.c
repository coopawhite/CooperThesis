    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>

    #define BUFFERSIZE 64
    int main(void){

        char *line = NULL, *token, *saveptr;
        size_t len = 0;
        ssize_t cmd = 0;

        printf("[INFO] C code is open and running... \n");

        line = (char *)malloc(BUFFERSIZE * sizeof(char));
        if(line == NULL){
            perror("Unable to allocate buffer");
            exit(1);
        }


        cmd = getline(&line, &len, stdin);
        printf("%s", line);
            //token = strtok_r(line, " ", &saveptr);
            
            //if (strcmp(token, "3") == 0){
            //printf("3");
            
       

        return(0);

    }
