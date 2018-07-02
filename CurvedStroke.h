#ifndef CURVEDSTROKE_H
#define CURVEDSTROKE_H

#include <iostream>
#include <vector>

class CurvedStroke {
public:
  CurvedStroke();
  ~CurvedStroke();
  int numControlPoints;
  int currControlPoint;
  int radius;
  void addControlPoint(float x, float y);
  std::vector<int> getNext();
  std::vector<std::vector<float>> controlPoints;
  int r, g, b;
};

#endif // CURVEDSTROKE_H
