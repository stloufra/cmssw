#include "DataFormats/Math/interface/SSEVec.h"
#ifdef  CMS_USE_SSE

#include "DataFormats/Math/interface/SSERot.h"
#include<cmath>
#include<iostream>

// this is a test,
using namespace mathSSE;

void addScaleddiff(Vec3F&res, float s, Vec3F const & a, Vec3F const & b) {
  res = res + s*(a-b);
} 

void addScaleddiffIntr(Vec3F&res, float s, Vec3F const & a, Vec3F const & b) {
  res.vec =  _mm_add_ps(res.vec, _mm_mul_ps(_mm_set1_ps(s), _mm_sub_ps(a.vec,b.vec)));
}


float dotV( Vec3F const & a, Vec3F const & b) {
  return dot(a,b);
}

float dotSimple( Vec3F const & a, Vec3F const & b) {
  Vec3F res = a*b;
  return res.arr[0]+res.arr[1]+res.arr[2];
}

double dotSimple( Vec3D const & a, Vec3D const & b) {
  Vec3D res = a*b;
  return res.arr[0]+res.arr[1]+res.arr[2];
}

float norm(Vec3F const & a) {
  return std::sqrt(dot(a,a));
}


Vec3F toLocal(Vec3F const & a, Rot3<float> const & r) {
  return r.rotate(a);
}

Vec3F toGlobal(Vec3F const & a, Rot3<float> const & r) {
  return r.rotateBack(a);
}



// fake basicVector to check constructs...
template<typename T>
struct BaVec { 
  typedef BaVec<T> self;

  BaVec() : 
    theX(0), theY(0), theZ(0), theW(0){}

  BaVec(float f1, float f2, float f3) : 
    theX(f1), theY(f2), theZ(f3), theW(0){}

  self & operator+=(self const & rh) {
    return *this;
  }

  T  theX; T  theY; T  theZ; T  theW;
}  __attribute__ ((aligned (16)));


typedef BaVec<float> BaVecF;

struct makeVec3F {
  makeVec3F(BaVecF & bv) : v(reinterpret_cast<Vec3F&>(bv)){}
  Vec3F & v;
};
struct makeVec3FC {
  makeVec3FC(BaVecF const & bv) : v(reinterpret_cast<Vec3F const&>(bv)){}
  Vec3F const & v;
};

template<>
inline BaVecF & BaVecF::operator+=(BaVecF const & rh) {
  makeVec3FC v(rh);
  makeVec3F s(*this);
  s.v = s.v + v.v;
  return *this;
}


void sum(BaVecF & lh, BaVecF const & rh) {
  lh += rh;  
}


void testBa() {
  std::cout <<" test BA" << std::endl;
  BaVecF vx(2.0,4.0,5.0);
  BaVecF vy(-3.0,2.0,-5.0);
  vx+=vy;
  std::cout << vx.theX << ", "<<  vx.theY << ", "<<  vx.theZ << std::endl;
}


void go2d() {

  typedef Vec2<double> Vec2d;
  typedef Vec3<double> Vec3d;

  std::cout << std::endl;
  std::cout << sizeof(Vec2d) << std::endl;

  Vec3d x(2.0,4.0,5.0);
  Vec3d y(-3.0,2.0,-5.0);
  std::cout << x << std::endl;
  std::cout << y << std::endl;

  Vec2d x2 = x.xy();
  Vec2d y2 = y.xy();
  std::cout << x2 << std::endl;
  std::cout << y2 << std::endl;


  std::cout << 3.*x2 << std::endl;
  std::cout << y2*0.1 << std::endl;
  std::cout << mathSSE::sqrt(x2) << std::endl;


  std::cout << dot(x2,y2) << " = 2?"<< std::endl; 
  

  double z = cross(x2,y2);
  std::cout << z  << " = 16?" << std::endl;

  std::cout <<  mathSSE::sqrt(z)  << " = 4?" << std::endl;

}

template<typename T> 
void go() {

  typedef Vec3<T> Vec;

  std::cout << std::endl;
  std::cout << sizeof(Vec) << std::endl;

  Vec x(2.0,4.0,5.0);
  Vec y(-3.0,2.0,-5.0);
  std::cout << x << std::endl;
  std::cout << -x << std::endl;
  std::cout << x.get1(2) << std::endl;
  std::cout << y << std::endl;
  std::cout << 3.*x << std::endl;
  std::cout << y*0.1 << std::endl;
  std::cout << (Vec(1) - y*0.1) << std::endl;
  std::cout <<  mathSSE::sqrt(x) << std::endl;


  std::cout << dot(x,y) << std::endl; 
  std::cout << dotSimple(x,y) << std::endl;

  std::cout << "equal" << (x==x ? " " : " not ") << "ok" << std::endl;
  std::cout << "not equal" << (x==y ? " not " : " ") << "ok" << std::endl;
 
  Vec z = cross(x,y);
  std::cout << z << std::endl;


  std::cout << "rotations" << std::endl;

  T a = 0.01;
  T ca = std::cos(a);
  T sa = std::sin(a);

  Rot3<T> r1( ca, sa, 0,
	      -sa, ca, 0,
	      0,  0, 1);

  Rot3<T> r2(Vec( 0, 1 ,0), Vec( 0, 0, 1), Vec( 1, 0, 0));

  Vec xr = r1.rotate(x);
  std::cout << x << std::endl;
  std::cout << xr << std::endl;
  std::cout << r1.rotateBack(xr) << std::endl;

  Rot3<T> rt = r1.transpose();
  Vec xt = rt.rotate(xr);
  std::cout << x << std::endl;
  std::cout << xt << std::endl;
  std::cout << rt.rotateBack(xt) << std::endl;

  std::cout << r1 << std::endl;
  std::cout << rt << std::endl;
  std::cout << r1*rt << std::endl;
  std::cout << r2 << std::endl;
  std::cout << r1*r2 << std::endl;
  std::cout << r2*r1 << std::endl;
  std::cout << r1*r2.transpose() << std::endl;
  std::cout << r1.transpose()*r2 << std::endl;

}


int main() {
  testBa();
  go<float>();
  go<double>();
  go2d();

  return 0;
};

#else
int main() {
  return 0;
}
#endif
