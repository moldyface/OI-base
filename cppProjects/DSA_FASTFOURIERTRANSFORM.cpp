#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll MAXN=8e5+5, MOD=998244353;
// credits @ adam.jq.xu
ll n, m, f[MAXN], aa[MAXN], bb[MAXN], cc[MAXN], dd[MAXN], ee[MAXN], ff[MAXN], gg[MAXN], hh[MAXN], ii[MAXN], inv, len, ord[MAXN];
// qpow
ll qpow(ll x, ll y){
  ll res=1;
  while(y){
    if(y&1){
      res=res*x%MOD;
    }
    x=x*x%MOD;
    y>>=1;
  }
  return res;
}
void NTT(ll *t, ll type){
	for(ll i=0; i<len; i++){
		if(i<ord[i]){
			swap(t[i], t[ord[i]]);
		}
	}
	for(ll i=1; i<len; i<<=1){
		ll x=qpow(type==1?3:332748118, (MOD-1)/(i<<1));
		for(ll j=0; j<len; j+=(i<<1)){
			for(ll k=j, l=1; k<j+i; k++, l=l*x%MOD){
				ll tmpa=t[k], tmpb=l*t[k+i]%MOD;
				t[k]=(tmpa+tmpb)%MOD;
				t[k+i]=(tmpa-tmpb+MOD)%MOD;
			}
		}
	}
}
void Multiply(ll *x, ll szx, ll *y, ll szy){
  len=1;
  while(len<=szx+szy){
    len<<=1;
  }
  ord[0]=0;
  for(ll i=0; i<len; i++){
    ord[i]=(ord[i>>1]>>1)|((i&1)?(len>>1):0);
  }
  NTT(x, 1);
  NTT(y, 1);
  for(ll i=0; i<len; i++){
    x[i]=x[i]*y[i]%MOD;
  }
  NTT(x, -1);
  inv=qpow(len, MOD-2);
  for(ll i=0; i<=len; i++){
    x[i]=x[i]*inv%MOD;
    y[i]=0;
  }
}
void Inverse(ll *x, ll szx){
  cc[0]=qpow(x[0],MOD-2);
  for(ll i=1; i<=szx; i<<=1){
    for(ll j=0; j<i; j++){
      aa[j]=cc[j];
    }
    for(ll j=0; j<i; j++){
      bb[j]=cc[j];
    }
    Multiply(aa,i-1,bb,i-1);
    for(ll j=0; j<(i<<1); j++){
      bb[j]=x[j];
    }
    Multiply(aa,(i-1)<<1,bb,(i<<1)-1);
    for(ll j=0; j<(i<<1); j++){
      cc[j]=(2*cc[j]+MOD-aa[j])%MOD;
    }
    for(ll j=0; j<(i<<2); j++){
      aa[j]=0;
    }
  }
  for(ll i=0; i<=(szx<<1); i++){
    x[i]=(i<=szx?cc[i]:0);
    cc[i]=0;
  }
}
void Ln(ll *x, ll szx){
  for(ll i=0; i<=szx; i++){
    dd[i]=x[i+1]*(i+1)%MOD;
  }
  Inverse(x, szx);
  Multiply(x, szx, dd, szx);
  for(ll i=(szx<<1); i>=1; i--){
    if(i<=szx) x[i]=x[i-1]*qpow(i,MOD-2)%MOD;
    else x[i]=0;
  }
  x[0]=0;
}
void Exp(ll *x, ll szx){
  ee[0]=1;
  for(ll i=1; i<=szx; i<<=1){
    for(ll j=0; j<i; j++){
      ff[j]=ee[j];
    }
    Ln(ff, i<<1);
    for(ll j=0; j<(i<<1); j++){
      ff[j]=((j==0)-ff[j]+x[j]+MOD)%MOD;
    }
    Multiply(ee,i-1,ff,(i<<1)-1);
    for(ll j=(i<<1); j<len; j++){
      ee[j]=0;
    }
  }
  for(ll i=0; i<=(szx<<1); i++){
    x[i]=(i<=szx?ee[i]:0);
    ee[i]=0;
  }
}
void Sqrt(ll *x, ll szx){
  gg[0]=1;
  for(ll i=1; i<=szx; i<<=1){
    for(ll j=0; j<i; j++){
      hh[j]=gg[j];
      ii[j]=gg[j]*2%MOD;
    }
    Multiply(gg,i-1,hh,i-1);
    for(ll j=0; j<(i<<1); j++){
      gg[j]=(gg[j]+x[j]+MOD)%MOD;
    }
    Inverse(ii,(i<<1));
    Multiply(gg,(i<<1),ii,(i<<1));
    for(ll j=(i<<1); j<len; j++){
      gg[j]=0;
    }
  }
  for(ll i=0; i<=(szx<<1); i++){
    x[i]=(i<=szx?gg[i]:0);
    gg[i]=0;
  }
}
void qpow(ll *x, ll szx, ll y){
  Ln(f,n);
  for(ll i=0; i<szx; i++){
    f[i]=f[i]*y%MOD;
  }
  Exp(f,n);
}
inline ll read(){
  char ch=getchar();ll x=0;
  while(ch<'0'||ch>'9'){ch=getchar();}
  while(ch>='0'&&ch<='9'){x=(x*10+ch-'0')%MOD;ch=getchar();}
  return x;
}
int main(){
  ios::sync_with_stdio(0);
  cin.tie(NULL);
  cout.tie(NULL);
  cin>>n;
  for(ll i=0; i<n; i++){
    cin>>f[i];
  }
  // find sqrt of function mod MOD
  // 0,1 : modular sqrt.
  Sqrt(f,n);
  for(ll i=0; i<n; i++){
    cout<<f[i]<<" ";
  }
  return 0;
}
