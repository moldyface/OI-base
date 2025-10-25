#include <bits/stdc++.h>
using namespace std;
typedef double ld;
typedef long long ll;
const ld epsilon = 1e-9;
inline int sgn(ld x) { return (x > epsilon) - (x < -epsilon); }
inline bool eq(ld a, ld b) { return sgn(a - b) == 0; }
// point stuff
// credits to @copilot
struct Point {
    ld x = 0, y = 0;
    Point(ld x = 0, ld y = 0) : x(x), y(y) {}
    Point operator+(const Point p)const { return Point(x + p.x, y + p.y); }
    Point operator-(const Point p)const { return Point(x - p.x, y - p.y); }
    Point operator*(const ld s)const { return Point(x * s, y * s); }
    Point operator/(const ld s)const { return Point(x / s, y / s); }
    bool operator==(const Point& p)const { return eq(x, p.x) && eq(y, p.y); }
    bool operator!=(const Point& p)const { return !(*this == p); }
    bool operator< (const Point& p)const {return (!eq(x,p.x) ? x < p.x : y < p.y - epsilon);}
    ld dist2() const { return x * x + y * y; }
    ld norm()const { return sqrt(dist2()); }
    ld dot(Point p)const { return x * p.x + y * p.y; }
    ld cross(Point p)const { return x * p.y - y * p.x; }
    Point unit()const { ld n = norm(); return n ? (*this) / n : *this; }
    Point perp()const { return Point(-y, x); }    
    Point rot(ld ang) const {                    
        ld c = cos(ang), s = sin(ang);
        return Point(c * x - s * y, s * x + c * y);
    }
};

struct Line {
    Point p1, p2;
    Line(Point p1, Point p2) : p1(p1), p2(p2) {}
};
inline ld cross(const Point a, const Point b, const Point c) { return (b - a).cross(c - a); }
inline ld dot(const Point a, const Point b,const  Point c) { return (b - a).dot(c - a); }
// Orientation: -1 left, 0 collinear, 1 right 
inline int turn(const Point a, const Point b, const Point c) {
    ld r = cross(a, b, c);
    int s = sgn(r);
    return(s < 0 ? 1 : (s == 0 ? 0 : -1));
}
// On-segment
inline bool onSegment(Point p, Line L) {
    if (sgn(cross(L.p1, L.p2, p)) != 0) return false;
    return sgn((p.x - L.p1.x) * (p.x - L.p2.x)) <= 0 && sgn((p.y - L.p1.y) * (p.y - L.p2.y)) <= 0;
}
inline bool segmentsIntersect(Point a, Point b, Point c, Point d) {
    ld c1 = cross(a, b, c), c2 = cross(a, b, d);
    ld c3 = cross(c, d, a), c4 = cross(c, d, b);
    int s1 = sgn(c1), s2 = sgn(c2), s3 = sgn(c3), s4 = sgn(c4);

    if (s1 * s2 < 0 && s3 * s4 < 0) return true; // proper crossing
    // endpoints / collinear overlaps
    return((s1 == 0 && onSegment(c, Line(a, b)))||(s2 == 0 && onSegment(d, Line(a, b)))||(s3 == 0 && onSegment(a, Line(c, d)))||(s4 == 0 && onSegment(b, Line(c, d))));
    return 0;
}

// Line-line intersection (a1-a2) with (b1-b2). Returns pair<hasIntersection, Point>
// If lines are parallel but collinear, returns false with undefined Point.
inline pair<bool, Point> lineIntersection(Point a1, Point a2, Point b1, Point b2) {
    Point r = a2 - a1, s = b2 - b1;
    ld rxs = r.cross(s);
    ld qpxr = (b1 - a1).cross(r);
    if (sgn(rxs) == 0) { // parallel
        if (sgn(qpxr) == 0) return {false, Point()}; // collinear: infinite intersections
        return {false, Point()}; // parallel disjoint
    }
    ld t = (b1 - a1).cross(s) / rxs;
    Point p = a1 + r * t;
    return {true, p};
}

// Projection of point p onto infinite line AB
inline Point projectPointOnLine(Point a, Point b, Point p) {
    Point ab = b - a;
    ld t = ab.dot(p - a) / max(ab.dist2(), epsilon);
    return a + ab * t;
}

// Projection of point p onto segment AB (clamped)
inline Point projectPointOnSegment(Point a, Point b, Point p) {
    Point ab = b - a;
    ld t = ab.dot(p - a) / max(ab.dist2(), epsilon);
    t = max(0.0, min(1.0, t));
    return a + ab * t;
}

inline ld distPointLine(Point a, Point b, Point p) {
    return fabs(cross(a, b, p)) / max((b - a).norm(), epsilon);
}
inline ld distPointSegment(Point a, Point b, Point p) {
    return (p - projectPointOnSegment(a, b, p)).norm();
}
inline ld distSegmentSegment(Point a, Point b, Point c, Point d) {
    if (segmentsIntersect(a, b, c, d)) return 0.0;
    return min({distPointSegment(a, b, c), distPointSegment(a, b, d),
                distPointSegment(c, d, a), distPointSegment(c, d, b)});
}

// Polygon area (signed; CCW positive), and absolute
inline ld polygonSignedArea(const vector<Point>& P) {
    long long n = (long long)P.size();
    ld s = 0;
    for (long long i = 0; i < n; ++i) {
        long long j = (i + 1) % n;
        s += P[i].cross(P[j]);
    }
    return 0.5 * s;
}
inline ld polygonArea(const vector<Point>& P) { return fabs(polygonSignedArea(P)); }

// Point-in-polygon: 0 outside, 1 on boundary, 2 inside (works for non-self-intersecting polygons)
inline int pointInPolygon(const vector<Point>& poly, Point q) {
    bool inside = false;
    int n = (int)poly.size();
    for (int i = 0, j = n - 1; i < n; j = i++) {
        Point a = poly[j], b = poly[i];
        if (onSegment(q, Line(a, b))) return 1;
        bool intersect = ((a.y > q.y) != (b.y > q.y)) &&
                         (q.x < (b.x - a.x) * (q.y - a.y) / (b.y - a.y + 0.0) + a.x);
        if (intersect) inside = !inside;
    }
    return inside ? 2 : 0;
}

// Convex hull (Monotone Chain). Returns CCW hull without repeating the first point at the end.
inline vector<Point> convexHull(vector<Point> P) {
    int n = (int)P.size();
    if (n <= 1) return P;
    sort(P.begin(), P.end());
    P.erase(unique(P.begin(), P.end()), P.end());
    vector<Point> lower, upper;
    for (auto& p : P) {
        while (lower.size() >= 2 && sgn(cross(lower[lower.size()-2], lower.back(), p)) <= 0)
            lower.pop_back();
        lower.push_back(p);
    }
    for (int i = (int)P.size() - 1; i >= 0; --i) {
        auto& p = P[i];
        while (upper.size() >= 2 && sgn(cross(upper[upper.size()-2], upper.back(), p)) <= 0)
            upper.pop_back();
        upper.push_back(p);
    }
    lower.pop_back(); upper.pop_back();
    lower.insert(lower.end(), upper.begin(), upper.end());
    return lower;
}

// Rotating calipers: diameter (farthest pair) on convex polygon (CCW, no duplicate end)


// ------------------------
// Tests and demos
// ------------------------
inline void testTurning() {
    Point P0(0, 0), P1(0, 3);
    cout << turn(P0, P1, Point(-1, 5)) << "\n"; // -1
    cout << turn(P0, P1, Point(0, 5)) << "\n";  // 0
    cout << turn(P0, P1, Point(1, 5)) << "\n";  // 1
}

inline void testOnSegment() {
    Point A(1, 0), B(2, 2), C(4, 6);
    Point X(5, 4), Y(3, 0), Z(6, 6);
    cout << onSegment(A, Line(B, C)) << "\n"; // 0
    cout << onSegment(B, Line(A, C)) << "\n"; // 1
    cout << onSegment(X, Line(Y, Z)) << "\n"; // 1
    cout << onSegment(Y, Line(Z, X)) << "\n"; // 0
    cout << onSegment(Y, Line(Y, Z)) << "\n"; // 1
}


int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    testTurning();
    testOnSegment();
    return 0;
}