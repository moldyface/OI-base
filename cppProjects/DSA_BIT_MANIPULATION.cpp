#include <bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll MAXN = 1e5;
const ll INF = 1e9;
inline void bit_operations(){
  ll x;
  cin >> x;
  if(x % 2 == 0){
    // this is slow!
    cout << "slow\n";
  }
  if(x & 1){
    // this is fast!
    cout << "fast\n";
  }

  if(x>0 && (x & (x-1)) == 0){
    // is x a power of 2?
    cout << "power of 2\n";
  }

  ll a = x & -x;
  // a is the lowest bit of x!
  // add context
  cout << "lowest bit: " << a << "\n";
  ll r = __builtin_ctz(x);
  // ending 0's of x
  cout << "ending 0's: " << r << "\n";
  ll c = __builtin_popcountll(x);
  // amnt of bits in 
  cout << "amnt of bits: " << c << "\n";
  ll u = __builtin_clz(x);
  // amnt of bits to the left of the first 1
  cout << "amnt of bits to the left of the first (int type) 1: " << u << "\n";
  ll l = __builtin_clzll(x);
  // amnt of bits to the left of the first 1
  cout << "amnt of bits to the left of the first (long long type)1: " << l << "\n";
  ll f = __builtin_ffs(x);
  // the lowest bit of x
  cout << "lowest bit: " << f << "\n";
  ll t = __builtin_ffsll(x);
  // the lowest bit of x
  cout << "lowest bit: " << t << "\n";

}
signed main() {
  ios::sync_with_stdio(0);
  cin.tie(0);
  cout.tie(0);
  bit_operations();
  return 0;
}