#include <stdio.h>
int hello(void) { 
  for(int i=0; i<5 ; i++)
    printf("Hello world, from CPU!\n");
  return 1;
}
int main(void) {
  if(hello()!=1) 
    printf("An error occured!\n");
  return 0;
}
