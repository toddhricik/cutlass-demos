*/#include <iostream>
#include <cute/layout.hpp>

/*
template <class Shape, class Stride>
void print1D(Layout<Shape,Stride> const& layout)
{
  for (int i = 0; i < size(layout); ++i) {
    printf("%3d  ", layout(i));
  }
}
template <class Shape, class Stride>
void print2D(Layout<Shape,Stride> const& layout)
{
  for (int m = 0; m < size<0>(layout); ++m) {
    for (int n = 0; n < size<1>(layout); ++n) {
      printf("%3d  ", layout(m,n));
    }
    printf("\n");
  }
  printf("\n");
}
*/
int main()
{
    using namespace cute;
    Layout s8 = make_layout(Int<8>{});
    print(s8);
/*
    Layout d8 = make_layout(8);

    Layout s2xs4 = make_layout(make_shape(Int<2>{},Int<4>{}));
    Layout s2xd4 = make_layout(make_shape(Int<2>{},4));

    Layout s2xd4_a = make_layout(make_shape (Int< 2>{},4),
                                make_stride(Int<12>{},Int<1>{}));
    Layout s2xd4_col = make_layout(make_shape(Int<2>{},4),
                                LayoutLeft{});
    Layout s2xd4_row = make_layout(make_shape(Int<2>{},4),
                                LayoutRight{});

    Layout s2xh4 = make_layout(make_shape (2,make_shape (2,2)),
                            make_stride(4,make_stride(2,1)));
    Layout s2xh4_col = make_layout(shape(s2xh4),
                                LayoutLeft{});

    print_layout(d8);
    print_layout(s2xs4);
    print_layout(s2xd4);
    print_layout(s2xd4_a);
    print_layout(s2xd4_col);
    print_layout(s2xd4_row);
    print_layout(s2xh4);
    print_layout(s2xh4_col);
  
    */
    std::cout << std::endl << "done" << std::endl;
    
}