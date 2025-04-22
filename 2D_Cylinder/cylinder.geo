// mesh size
h = .0165;

// points: corners
Point(1) = {0, 0, 0, h};
Point(2) = {0, 1, 0, h};
Point(3) = {.5, 0, 0, h};
Point(4) = {.5, 1, 0, h};

// points: cylinder
Point(5) = {.5, .65, 0, h};
Point(6) = {.5, .35, 0, h};
Point(7) = {.5, .5, 0, h};

// define semicircular cylinder
Circle(8) = {5, 7, 6};

// lines to define edges of domain
Line(9) = {1, 3};
Line(10) = {3, 6}; 
Line(11) = {5, 4};
Line(12) = {4, 2}; 
Line(13) = {2, 1};

Line Loop(14) = {9,10,-8,11,12,13};

Plane Surface(15) = {14};
Mesh.MshFileVersion = 2.2;
Mesh 2;
