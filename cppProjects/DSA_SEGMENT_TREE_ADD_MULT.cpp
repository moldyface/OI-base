#include<bits/stdc++.h>
using namespace std;
typedef long long ll;
const ll MAXN = 1e5;
ll MOD;
vector<ll> a(MAXN);
ll tree[MAXN];
vector<ll> lazy1(MAXN, 1);
vector<ll> lazy2(MAXN, 0);

void build(ll id, ll l, ll r) {
    if (l == r) {
        tree[id] = a[l];
        return;
    }
    ll mid = l + (r - l) / 2;
    build(id<<1, l, mid);
    build(id<<1|1, mid + 1, r);
    tree[id] = (tree[id<<1] + tree[id<<1|1])%MOD;
}

void push_down(ll id, ll L, ll mid, ll R) {
    ll len_left = mid - L + 1;
    ll len_right = R - mid;
    tree[id<<1] = tree[id<<1] * lazy1[id] + lazy2[id] * len_left;
    tree[id<<1|1] = tree[id<<1|1] * lazy1[id] + lazy2[id] * len_right;

    lazy1[id<<1] = lazy1[id] * lazy1[id<<1];
    lazy2[id<<1] = lazy2[id<<1] * lazy1[id] + lazy2[id];

    lazy1[id<<1|1] = lazy1[id] * lazy1[id<<1|1];
    lazy2[id<<1|1] = lazy2[id<<1|1] * lazy1[id] + lazy2[id];

    lazy1[id] = 1;
    lazy2[id] = 0;
}

ll query(ll id, ll L, ll R, ll QL, ll QR) {
    if (QR < L || R < QL)
        return 0;
    if (QL <= L && R <= QR)
        return tree[id];
    ll mid = L + (R - L) / 2;
    push_down(id, L, mid, R);
    return (query(id<<1, L, mid, QL, QR) + 
            query(id<<1|1, mid + 1, R, QL, QR))%MOD;
}

void update_range_add(ll id, ll L, ll R, ll QL, ll QR, ll val) {
    if (QR < L || R < QL)
        return;
    if (QL <= L && R <= QR) {
        tree[id] += (val * (R - L + 1))%MOD;
        lazy2[id] += val;
        return;
    }
    ll mid = L + (R - L) / 2;
    push_down(id, L, mid, R);
    update_range_add(id<<1, L, mid, QL, QR, val);
    update_range_add(id<<1|1, mid + 1, R, QL, QR, val);
    tree[id] = (tree[id<<1] + tree[id<<1|1])%MOD;;
}

void update_range_mlt(ll id, ll L, ll R, ll QL, ll QR, ll val) {
    ll mid = L + (R - L) / 2;
    if (QR < L || R < QL)
        return;
    if (QL <= L && R <= QR) {
        tree[id] = (tree[id] * val) % MOD;
        lazy1[id] = lazy1[id] * val % MOD;
        lazy2[id] = lazy2[id] * val % MOD;
        return;
    }
    push_down(id, L, mid, R);
    update_range_mlt(id<<1, L, mid, QL, QR, val);
    update_range_mlt(id<<1|1, mid + 1, R, QL, QR, val);
    tree[id] = (tree[id<<1] + tree[id<<1|1])%MOD;
}

void tree_prll(ll n) {
    ll h = ceil(log2(n)) + 1;
    for (ll i = 0; i < h; i++) {
        ll start = (1 << i);
        ll end = (1 << (i + 1)) - 1;
        for (ll j = start; j <= end && j < MAXN; j++) {
            cout << tree[j] << " ";
        }
        cout << endl;
    }
}

signed main() {
    ll c, s;
    cin >> c >> s >> MOD;
    for (ll i = 1; i <= c; i++)
        cin >> a[i];
    build(1, 1, c);
    while (s--) {
        ll op;
        cin >> op;
      //tree_prll(c);
        if (op == 1) {
            ll l, r, val;
            cin >> l >> r >> val;
            update_range_mlt(1, 1, c, l, r, val);
        } 
      else if (op == 2) {
            ll l, r, val;
            cin >> l >> r >> val;
            update_range_add(1, 1, c, l, r, val);
        } 
      else{
            ll l, r;
            cin >> l >> r;
            ll res = query(1, 1, c, l, r);
           cout << res % MOD << "\n";
        }
    }
    return 0;
}