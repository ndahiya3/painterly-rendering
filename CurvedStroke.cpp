#include "CurvedStroke.h"

CurvedStroke::CurvedStroke() {
  numControlPoints = 0;
  currControlPoint = 0;
  r = g = b = 0;
  radius = 0;
}

CurvedStroke::~CurvedStroke() { controlPoints.clear(); }

void CurvedStroke::addControlPoint(float x, float y) {
  std::vector<float> pt;
  pt.push_back(x);
  pt.push_back(y);
  controlPoints.push_back(pt);
  numControlPoints++;
}

std::vector<int> CurvedStroke::getNext() {}
