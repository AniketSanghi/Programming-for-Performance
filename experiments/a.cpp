#include <bits/stdc++.h>
using namespace std;

#define ll long long int
#define pb push_back
#define rz resize
#define fst first
#define snd second
#define mkp make_pair
#define rep(i,n) for(ll i=0; i<n; ++i)
#define repi(i,a,n) for(ll i=a; i<n; ++i)
#define repii(i,a,n,b) for(ll i=a; i<n; i+=b)

// Defining constants
#define mod 1000000007
#define inf 100000000000

// Defining Functions
#define min(a,b) (a<b?a:b)
#define max(a,b) (a>b?a:b)

// Long Types likhne main no time waste
typedef vector <ll> vll;
typedef vector <int> vi;


// Some Number theory functions
void swap(ll &a, ll &b) {
	ll temp = a;
	a = b;
	b = temp;
}

ll power(ll a, ll b) {
	ll ans = 1;
	while(b) {
		if(b%2) ans = (ans*a)%mod;
		a = (a*a)%mod;
		b = b>>1;
	}
	return ans;
}

ll gcd(ll a, ll b)
{
	if(!b) return a;
	if(a<b) swap(a, b);
	return gcd(b,a%b);
}


// Things used with grpah
struct node {
    ll a,b;
};
typedef vector <node> vn;
vector <vll> graph;

ll cube(ll a) {
	return a*a%mod*a%mod;
}

int main()
{
	ll a = 0;
	rep(i,1000000000) a = (a + cube(i))%mod;
    cout<<a;
    return 0;
}