#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
#define all(x) x.begin(),x.end()
const ll MAXN = 1e5;
const ll MOD = 1000000007;
mt19937 mt(time(nullptr));
vector<ll> primes = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199};

ll power(ll a, ll n, ll p) {
  ll temp = 1;
  while (n) {
      if (n & 1) temp = temp * a % p;
      a = a*a % p, n >>= 1;
  }
  return temp;
}

// harmonic Division.
ll val_sqrtn(ll k){
    ll bound = sqrt(k-1)+1;
    ll res = 0;
    for(ll i = 1; i < bound; i++){
        res += (k/i);
        res %= MOD;
    }
    ll nxt = k/bound;
    for(ll i = nxt; i ; i--){
        ll top = min(k/i,k);
        ll bottom = k/(i+1) + 1;
        res += (top-bottom+1)*(i);
        res%=MOD;
    }
    return res;
}
//trial Division - prime detection
// O(sqrt(n))
inline bool prem(ll b){
  if(b == 1) return false;
  ll ans = sqrt(b);
  for(ll i = 2; i <= ans; i++){
    if(b % i == 0 ) return false;
  }
  return true;
}



//Miller rabin primality test
// O(log(n))
// probablistic
inline bool milr(ll a, ll b){
  ll c = 1;
  ll d = b-1;
  if(power(a,d,b) != 1){ return 0;}
  while(!d&1){
    ll e= power(a,d,b);
    if(c == 1){
      if(!(e == b-1 || e == 1)) return 0;
    }
    c = e;
    d >>= 1;
  }
  return 1;
}
inline bool upr(ll b){
  for(ll c : primes){
    if(!(b%c)){
      if(b == c){
        return true;
      }
      return false;
    }
  }
  return true;
}
inline bool cmilr(ll b){
  if(b == 1) return false;
  if(b == 2) return true;
  if(!upr(b)) return false;
  ll a = mt();
  a %= b;
  a = max(a,2LL);
  if(!milr(a,b)){
    return false;
  }
  a = mt();
  a %= b;
  a = max(a,2LL);
  if(!milr(a,b)){
    return false;
  }
  a = mt();
  a %= b;
  a = max(a,2LL);
  if(!milr(a,b)){
    return false;
  }
  return true;
}

// Sieves:

// till limit using Sieve of Atkin
// O(n) space, time complexity
vector<ll> sieveoa(ll limit) {

    // llialise the arr array
    // with initial 0 values
    vector<ll> arr(limit + 1);

    // mark 2 and 3 as prime
    if (limit > 2) arr[2] = 1;
    if (limit > 3) arr[3] = 1;

    // check for all three conditions
    for(ll x = 1; x * x <= limit; x++) {
        for(ll y = 1; y * y <= limit; y++) {
            ll x2 = x*x;
            ll y2 = y*y;
            // condition 1
            ll n = (4 * x2) + (y2);
            if(n <= limit && (n % 12 == 1 || n % 12 == 5)) 
                arr[n] = (arr[n] + 1) % 2;

            // condition 2
            n = (3 * x2) + (y2);
            if(n <= limit && n % 12 == 7) 
                arr[n] = (arr[n] + 1) % 2;

            // condition 3
            n = (3 * x2) - (y2);
            if(x > y && n <= limit && n % 12 == 11) 
                arr[n] = (arr[n] + 1) % 2;
        }
    }

    // Mark all multiples
    // of squares as non-prime
    for (ll i = 5; i * i <= limit; i++) {
        if (arr[i] == 0) continue;
        for (ll j = i * i; j <= limit; j += i * i)
            arr[j] = 0;
    }

    // store all prime numbers
    vector<ll> primes;
    for (ll i = 2; i <= limit; i++) {
        if (arr[i] == 1) {
            primes.push_back(i);
        }
    }
    return primes;
}
// miller rabin sieve - good for TC
vector<ll> sieve(ll a,ll b){
    vector<ll> res = {};
    for(ll i = 1; i <= b; i++){
      if(cmilr(i)) res.push_back(i);
    }
    return res;
  }


// sieve of eras.
vector<ll> sievee(ll n) {
    // creation of boolean array
    vector<bool> prime(n + 1, true);
    for (ll p = 2; p * p <= n; p++) {
        if (prime[p] == true) {
            
            // marking as false
            for (ll i = p * p; i <= n; i += p)
                prime[i] = false;
        }
    }
    vector<ll> res;
    for (ll p = 2; p <= n; p++){
        if (prime[p]){ 
            res.push_back(p);
        }
    }
    return res;
}

// prime factorization
vector<ll> pf(ll m){
    ll i = 2;
    vector<ll> res = {};
    while(m!=1){
      if(!(m%i)){
        res.push_back(i);
        m/=i;
      }
      else i++;
    }
    return res;
  }

// modular inverse.

inline ll modularinverse(ll a, ll p){
  return power(a,(p-2),p);
}
  


// akin euclidean.
//https://en.oi-wiki.org/math/euclidean/
inline ll solve(ll a, ll b, ll c, ll n) {
  ll n2 = n * (n + 1) / 2;
  if (a >= c || b >= c)
    return solve(a % c, b % c, c, n) + (a / c) * n2 + (b / c) * (n + 1);
  ll m = (a * n + b) / c;
  if (!m) return 0;
  return m * n - solve(c, c - b - 1, a, m - 1);
}

inline int akin_euclidean() {
  ll t;
  cin >> t;
  while(t--){
    ll a, b, c, n;
    cin >> n >> c >> a >> b;
    cout << solve(a, b, c, n - 1) << '\n';
  }
  return 0;
}




// binomial.
ll fact[MAXN], facti[MAXN], t[MAXN];

void binom_init(ll n){
  fact[0] = 1;
  for(ll i = 1; i <= n; ++i) {fact[i] = fact[i-1]*i; fact[i] %= MOD;}
  facti[n] = modularinverse(fact[n],MOD);
  for(ll i = n-1; i>= 0; --i){
    facti[i] = facti[i+1]*(i+1);
    facti[i] %= MOD;
  }
}
ll binom(ll n,ll m){
  binom_init(max(n,m));
  if(n < 0LL || m < 0LL || n < m ) { return 0LL; }
  return ((fact[n] * facti[m])%MOD * facti[n-m])%MOD;
}




// euler totient function
inline int phi(int n) {
  int result = n;
  for (int i = 2; i * i <= n; i++) {
      if (n % i == 0) {
          while (n % i == 0)
              n /= i;
          result -= result / i;
      }
  }
  if (n > 1)
      result -= result / n;
  return result;
}


// self explanatory
ll sum_of_divisors_from_one_to_n_in_sqrt_time_complexity(ll n){
  ll ans = 0;
  for(ll i = 1,r; i <= n; i = r+1){
    ll sr = n/i;
    if(sr) r=min(n/sr,n);
    else break;
    ll arith =sr * (r-i+1) * (i+r);
    arith>>=1;
    ans += arith;
  }
  return ans;
}


ll linear_sieve_for_phi(){
  vector<ll> phi(MAXN + 1, 1), sieve(MAXN + 1, -1), prefphi(MAXN + 1, 0);
    for(ll i = 2; i <= MAXN; i++)
    if(sieve[i] == -1){
      sieve[i] = i;
      for(ll j = i * i; j <= MAXN; j += i)
        sieve[j] = i;
    }

  for(ll i = 2; i <= MAXN; i++){
    ll p = sieve[i], j = i;
    while(j % p == 0){
      phi[i] *= p;
      j /= p;
    }
    phi[i] = (phi[i] / p) * (p - 1) * phi[j];
  }
  prefphi[0] = 0;
  for(ll i = 1; i <= MAXN; i++){
    prefphi[i] = prefphi[i-1]+phi[i];
  }
  return 0;
}

ll sumofnumbers(ll a, ll b){
    return ((a+b)*(b-a + 1) >> 1ll);
}
ll sumofsquares(ll a){
    return (a* (a+1) * (2*a+1)) / 6;
}
ll sumofoddsquares(ll a){
    return sumofsquares(a) - 4*sumofsquares(a / 2);
}
vector<ll> digit(ll val){
    vector<ll> sh;
    while(val){
        sh.push_back(val%10);
        val /= 10;
    }
    reverse(all(sh));
    return sh;
}
int main(){
    ll res = 0;
    for(ll i = 100; i <= 999; i++)
    for(ll j = 100; j <= 999; j++){
        ll val = i*j;
        vector<ll> d = digit(val);
        vector<ll> k = d;
        reverse(all(d));
        if(d == k){
            res = max(res, val);
        }
    }
    cout << res;
}