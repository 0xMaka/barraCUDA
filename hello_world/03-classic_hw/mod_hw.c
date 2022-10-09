#include <stdio.h>
int hello(void) {
  int n = 0;
  for(int i=0; i<5; i++) {
    printf("Hello world, from CPU!\n");
    printf("I am iteration %d\n", n);
    n++;
  }
  return 1;
}
int main(void) {
  if(hello()!=1) 
    printf("An error occured!\n");
  return 0;
}

/* ex. Output
Hello world, from CPU!
I am iteration 0
Hello world, from CPU!
I am iteration 1
Hello world, from CPU!
I am iteration 2
Hello world, from CPU!
I am iteration 3
Hello world, from CPU!
I am iteration 4
*/
