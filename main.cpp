#include "CurvedStroke.h"
#include <algorithm>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <random>
#include <vector>

using namespace std;
using namespace cv;
float calcAreaError(int x, int y, int gridSize, const Mat &I_orig,
                    const Mat &I_ref, int &i_argmax, int &j_argmax);
void paintCircularStroke(int x, int y, int radius, Mat &I_canvas,
                         const Mat &I_ref);
bool cubicSpline(const vector<vector<float>> &cntrlPts,
                 vector<vector<float>> &curvePts);
void getBasisFunctions(float u, float *splineWeights);
void paintSpline(vector<CurvedStroke *> &splineStrokes, Mat &I_canvas);
void paintCircularStrokeSpline(float x, float y, int radius, Mat &I_canvas,
                               const Vec3b &color);

int main() {
  string fileName;
  int brushType1;

  cout << "Please enter filename to process: ";
  getline(cin, fileName);
  cout << "Please select circular or curved strokes; 0==circular, 1==curved : ";
  cin >> brushType1;

  Mat I_orig = imread(fileName);

  int height = I_orig.rows;
  int width = I_orig.cols;

  int brushRadii[3];
  int numBrushes = 3;
  float f_sigma;
  float errorThreshold;
  int minStrokeLen;
  int maxStrokeLen;
  float fc;
  int brushType;

  for (int i = 0; i < 5; i++) {
    if (i == 0) {
      errorThreshold = 100;
      brushRadii[0] = 8;
      brushRadii[1] = 4;
      brushRadii[2] = 2;
      fc = 1.0;
      f_sigma = 0.5;
      minStrokeLen = 4;
      maxStrokeLen = 16;
      brushType = brushType1;
    } else if (i == 1) {
      errorThreshold = 50;
      brushRadii[0] = 8;
      brushRadii[1] = 4;
      brushRadii[2] = 2;
      fc = 0.25;
      f_sigma = 0.5;
      minStrokeLen = 10;
      maxStrokeLen = 16;
      brushType = brushType1;
    } else if (i == 2) {
      errorThreshold = 200;
      brushRadii[0] = 8;
      brushRadii[1] = 4;
      brushRadii[2] = 2;
      fc = 1.0;
      f_sigma = 0.5;
      minStrokeLen = 4;
      maxStrokeLen = 16;
      brushType = brushType1;
    } else if (i == 3) {
      errorThreshold = 100;
      brushRadii[0] = 0;
      brushRadii[1] = 4;
      brushRadii[2] = 2;
      fc = 1.0;
      f_sigma = 0.5;
      minStrokeLen = 0;
      maxStrokeLen = 0;
      brushType = 0;
    } else if (i == 4) {
      errorThreshold = 50;
      brushRadii[0] = 8;
      brushRadii[1] = 4;
      brushRadii[2] = 2;
      fc = 0.5;
      f_sigma = 0.5;
      minStrokeLen = 10;
      maxStrokeLen = 16;
      brushType = brushType1;
    }
    // Create empty canvas, Reference image (blurred image),
    // luminance image, gradient magnitudes images
    Mat canvas(I_orig.size(), I_orig.type(), cvScalar(0, 0, 0));
    Mat I_ref(I_orig.size(), I_orig.type());
    Mat I_luminance(height, width, CV_32FC1);
    Mat dx(height, width, CV_32FC1);
    Mat dy(height, width, CV_32FC1);

    // For each brush radii
    bool refresh = true; // Initially paint all gridpoints
    vector<vector<int>> strokes;
    vector<CurvedStroke *> CubicSpleenStrokes;

    for (int brush = 0; brush < numBrushes; brush++) {
      int brushRadius = brushRadii[brush];
      if (brushRadius == 0)
        continue;
      // Create gaussian blurred image fsig*Ri
      GaussianBlur(I_orig, I_ref, Size(0, 0), f_sigma * brushRadius);

      // Create luminance image and calculate sobel gradient
      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          Vec3b pix = I_ref.at<Vec3b>(y, x); // BGR
          float luminance_val =
              0.3f * pix.val[2] + 0.59f * pix.val[1] + 0.11f * pix.val[0];
          I_luminance.at<float>(y, x) = luminance_val / 255.0f;
        }
      }
      Sobel(I_luminance, dx, CV_32FC1, 1, 0, 3);
      Sobel(I_luminance, dy, CV_32FC1, 0, 1, 3);

      // For each grid (x,y) calc error and Max error location
      int gridSize = f_sigma * brushRadius;
      strokes.clear();
      CubicSpleenStrokes.clear();

      for (int y = 0; y < height; y += gridSize) {
        for (int x = 0; x < width; x += gridSize) {
          int i, j;
          float error = calcAreaError(x, y, gridSize, canvas, I_ref, i, j);
          // If error > T apply brush stroke with Ri
          if (error >= errorThreshold || refresh) {
            // Circular stroke coordinates
            vector<int> currStroke;
            currStroke.push_back(i);
            currStroke.push_back(j);
            strokes.push_back(currStroke);
            CurvedStroke *K = new CurvedStroke();
            K->addControlPoint(i, j);
            Vec3b color = I_ref.at<Vec3b>(j, i); // BGR
            K->r = color.val[2];
            K->g = color.val[1];
            K->b = color.val[0];
            K->radius = brushRadius;

            float x_i_minus_1 = i;
            float y_i_minus_1 = j;
            float x_i = i;
            float y_i = j;
            float del_x_i = 0, del_y_i = 0, del_x_i_minus_1 = 0,
                  del_y_i_minus_1 = 0;
            for (int k = 1; k <= maxStrokeLen; k++) {
              // Cubic spline stroke control points
              // Derivatives at (i,j)
              float gx =
                  255.0f * dx.at<float>(round(y_i_minus_1), round(x_i_minus_1));
              float gy =
                  255.0f * dy.at<float>(round(y_i_minus_1), round(x_i_minus_1));
              if (brushRadius * sqrt(gx * gx + gy * gy) >= 1.0f) {
                del_x_i = -gy;
                del_y_i = gx;
                if (k > 1 && ((del_x_i_minus_1 * del_x_i +
                               del_y_i_minus_1 * del_y_i) < 0)) {
                  del_x_i = -del_x_i;
                  del_y_i = -del_y_i;
                }
                // Filter stroke direction
                if (k > 1) {
                  del_x_i = fc * del_x_i + (1.0f - fc) * del_x_i_minus_1;
                  del_y_i = fc * del_y_i + (1.0f - fc) * del_y_i_minus_1;
                }
              } else {
                if (k > 1) {
                  del_x_i = del_x_i_minus_1;
                  del_y_i = del_y_i_minus_1;
                } else {
                  break; // return K;
                }
              }
              x_i =
                  x_i_minus_1 + brushRadius * del_x_i /
                                    sqrt(del_x_i * del_x_i + del_y_i * del_y_i);
              y_i =
                  y_i_minus_1 + brushRadius * del_y_i /
                                    sqrt(del_x_i * del_x_i + del_y_i * del_y_i);

              // Control points should be within grid
              if ((x_i < 0 || x_i > width - 1) || (y_i < 0 || y_i > height - 1))
                break;

              if (k > minStrokeLen) {
                Vec3b ref_color = I_ref.at<Vec3b>(round(y_i), round(x_i));
                Vec3b canvas_color = canvas.at<Vec3b>(round(y_i), round(x_i));
                Vec3f ref_canvas_diff = (Vec3f)ref_color - (Vec3f)canvas_color;
                float diff1 =
                    sqrt(ref_canvas_diff.val[0] * ref_canvas_diff.val[0] +
                         ref_canvas_diff.val[1] * ref_canvas_diff.val[1] +
                         ref_canvas_diff.val[2] * ref_canvas_diff.val[2]);
                Vec3f ref_stroke_diff = (Vec3f)ref_color - (Vec3f)color;
                float diff2 =
                    sqrt(ref_stroke_diff.val[0] * ref_stroke_diff.val[0] +
                         ref_stroke_diff.val[1] * ref_stroke_diff.val[1] +
                         ref_stroke_diff.val[2] * ref_stroke_diff.val[2]);
                if (diff1 < diff2)
                  break; // return K
              }
              K->addControlPoint(x_i, y_i);
              x_i_minus_1 = x_i;
              y_i_minus_1 = y_i;
              del_x_i_minus_1 = del_x_i;
              del_y_i_minus_1 = del_y_i;
            }
            CubicSpleenStrokes.push_back(K);
          }
        }
      }
      // cout << CubicSpleenStrokes.size() << "\t" << strokes.size() << endl;

      // Paint strokes randomly
      if (brushType == 0) {
        random_shuffle(strokes.begin(), strokes.end());
        for (vector<vector<int>>::iterator it = strokes.begin();
             it != strokes.end(); ++it)
          paintCircularStroke((*it)[0], (*it)[1], brushRadius, canvas, I_ref);
      } else if (brushType == 1) {
        random_shuffle(CubicSpleenStrokes.begin(), CubicSpleenStrokes.end());
        paintSpline(CubicSpleenStrokes, canvas);
      } else {
        random_shuffle(strokes.begin(), strokes.end());
        for (vector<vector<int>>::iterator it = strokes.begin();
             it != strokes.end(); ++it)
          paintCircularStroke((*it)[0], (*it)[1], brushRadius, canvas, I_ref);
      }
      //            if(brush==0)
      //                imwrite("layer1.png",canvas);
      //            else if(brush==1)
      //                imwrite("layer2.png", canvas);
      //            else
      //                imwrite("layer3.png",canvas);

      refresh = false;
    }
    string saveFileName;
    if (i == 0)
      saveFileName = "Impressionist.png";
    else if (i == 1)
      saveFileName = "Expresionist.png";
    else if (i == 2)
      saveFileName = "ColoristWash.png";
    else if (i == 3)
      saveFileName = "Pointilist.png";
    else if (i == 4)
      saveFileName = "Psychedelic.png";
    imwrite(saveFileName, canvas);
  }
  cout << "Done Processing\n";
  return 0;
}

float calcAreaError(int x, int y, int gridSize, const Mat &I_orig,
                    const Mat &I_ref, int &i_argmax, int &j_argmax) {
  // Calculate area error in gridSize grid and location where max error occurs
  float areaError = 0;
  float maxError = -99999999999;
  for (int row = y - gridSize / 2; row <= y + gridSize / 2; row++) {
    for (int col = x - gridSize / 2; col <= x + gridSize / 2; col++) {
      Vec3b bgrpixelOrig = I_orig.at<Vec3b>(row, col);
      Vec3b bgrpixelRef = I_ref.at<Vec3b>(row, col);
      Vec3f diff = (Vec3f)bgrpixelOrig - (Vec3f)bgrpixelRef;
      float error = sqrt(diff.val[0] * diff.val[0] + diff.val[1] * diff.val[1] +
                         diff.val[2] * diff.val[2]);
      areaError += error;
      if (error > maxError) {
        i_argmax = col;
        j_argmax = row;
      }
    }
  }

  return areaError / (gridSize * gridSize);
}

void paintCircularStroke(int x, int y, int radius, Mat &I_canvas,
                         const Mat &I_ref) {
  // Paint a circular stroke at location x,y on canvas image using reference
  // image
  float radiusSquare = radius * radius;
  int height = I_ref.rows;
  int width = I_ref.cols;
  Vec3b color = I_ref.at<Vec3b>(y, x);

  for (int row = y - radius; row <= y + radius; row++) {
    for (int col = x - radius; col <= x + radius; col++) {

      if (((col - x) * (col - x) + (row - y) * (row - y)) <= radiusSquare) {
        if ((col >= 0 && col < width) && (row >= 0 && row < height)) {
          I_canvas.at<Vec3b>(row, col) = color;
        }
      }
    }
  }
}

void paintSpline(vector<CurvedStroke *> &splineStrokes, Mat &I_canvas) {

  for (int i = 0; i < splineStrokes.size(); i++) {
    CurvedStroke *currStroke = splineStrokes.at(i);
    vector<vector<float>> curvePts;
    bool success = cubicSpline(currStroke->controlPoints, curvePts);
    if (success) {
      Vec3f color;
      color.val[0] = currStroke->b;
      color.val[1] = currStroke->g;
      color.val[2] = currStroke->r;
      int brushRadius = currStroke->radius;

      for (int j = 0; j < curvePts.size(); j++) {
        float x = curvePts.at(j).at(0);
        float y = curvePts.at(j).at(1);
        paintCircularStrokeSpline(x, y, brushRadius, I_canvas, color);
      }
    }
  }
}

void paintCircularStrokeSpline(float x, float y, int radius, Mat &I_canvas,
                               const Vec3b &color) {
  // Paint a circular stroke with center in between grid points
  // Used for painting spline segments as dense circular strokes

  int l = floor(x - radius);
  int r = floor(x + radius);
  int u = floor(y - radius);
  int d = floor(y + radius);
  int height = I_canvas.rows;
  int width = I_canvas.cols;
  float radiusSq = radius * radius;

  for (int row = u; row <= d; row++)
    for (int col = l; col <= r; col++)
      if ((row >= 0 && row < height) && (col >= 0 && col < width))
        if (((col - x) * (col - x) + (row - y) * (row - y)) <= radiusSq)
          I_canvas.at<Vec3b>(row, col) = color;
}

bool cubicSpline(const vector<vector<float>> &cntrlPts,
                 vector<vector<float>> &curvePts) {
  // Given set of control points return set of curve points for the
  // corresponding cubic spline

  int numPts = cntrlPts.size();
  if (numPts < 4)
    return false;
  int numSegments = numPts - 4 + 1;
  float numSteps = 10;
  float incr = 1.0f / numSteps;
  float *splineWeights = new float[4];

  for (int i = 0; i < numSegments; i++) {
    // For each curve segments
    float u = 0;
    while (u <= 1.0f) {
      getBasisFunctions(u, splineWeights);
      float Qx = 0, Qy = 0;
      for (int k = 0; k < 4; k++) {
        Qx += splineWeights[k] * cntrlPts[i + k][0];
        Qy += splineWeights[k] * cntrlPts[i + k][1];
      }
      vector<float> currCurvPt;
      currCurvPt.push_back(Qx);
      currCurvPt.push_back(Qy);
      curvePts.push_back(currCurvPt);
      u += incr;
    }
  }
  delete[] splineWeights;
  return true;
}

void getBasisFunctions(float u, float *splineWeights) {

  float uSq = u * u;
  float uCube = uSq * u;

  splineWeights[0] = (1.0f - 3.0f * u + 3.0f * uSq - uCube) / 6.0f;
  splineWeights[1] = (3.0f * uCube - 6.0f * uSq + 4.0f) / 6.0f;
  splineWeights[2] = (-3.0f * uCube + 3.0f * uSq + 3.0f * u + 1.0f) / 6.0f;
  splineWeights[3] = uCube / 6.0f;
}
