//Author: Martina Boscolo Bacheto
#ifndef MAIN_H
#define MAIN_H

#include "ball_classification.h"
#include "Balls_segmentation.h"
#include "Classes.h"
#include "Field_detection.h"
#include "Metrics.h"
#include "table_orientation.h"
#include "tracking.h"
#include "utilities.h"


void main_field_balls();
int main_tracker();
void main_metrics();

#endif // MAIN_H