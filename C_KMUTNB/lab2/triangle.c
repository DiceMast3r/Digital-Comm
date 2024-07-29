#include <stdio.h>

int main() {
    int a, b, c;
    printf("Input three sides of a triangle:\n");
    scanf("%d %d %d", &a, &b, &c);
    if (a + b > c && a + c > b && b + c > a) {
        if (a == b && b == c) {
            printf("This is a equilateral triangle\n");
        } else if (a == b || b == c || a == c) {
            printf("This is an isosceles triangle\n");
        } else {
            printf("This is a scalene triangle\n");
        }
    } else {
        printf("Not a triangle\n");
    }

    return 0;
};
