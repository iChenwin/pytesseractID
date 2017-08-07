#include <assert.h>
#include <stdio.h>

char *stringcpy(char *dest, const char *src) {

    assert((dest != NULL) && (src != NULL));

    char *address = dest;
    while ((*dest++ = *src++) != '\0') {

    }

    return address;
}

unsigned int stringlen(const char *str) {
    assert(str != NULL);
    
    unsigned int cnt = 0;

    while(*str++ != '\0') {
        cnt++;
    }

    return cnt;
}

int main(void) {
    char *a = "hello world!";
    char b[] = {0};

    char *c = stringcpy(b, a);

    printf("return:%s, b:%s, len:%d \n", c, b, stringlen(c));

    return 0;
}