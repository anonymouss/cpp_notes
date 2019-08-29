#include "CurrentConditionsDisplay.h"
#include "StatisticsDisplay.h"
#include "WeatherData.h"

#include <iostream>
#include <memory>

int main() {
    auto station = std::make_shared<WeatherData>();

    CurrentConditionsDisplay currentBoard;
    StatisticsDisplay statBoard;

    currentBoard.attach(station);
    statBoard.attach(station);

    station->setMeasurements(80.0f, 65.0f, 30.4f);

    currentBoard.detach();

    station->setMeasurements(82.0f, 70.0f, 29.2f);

    statBoard.detach();

    station->setMeasurements(78.0f, 90.0f, 29.2f);
}

/**
 * outputs
 * INFO: new observer registered. total :1
 * INFO: new observer registered. total :2
 * INFO: station is updating measurements... will notify observers...
 * <<-- Current Conditions Display Board -->>
 *   : current temperature : 80F degrees.
 *   : current    humidity : 65% humidity.
 * =========================================
 * <<-- Statistics Display Board -->>
 *   : average temperature : 80F degrees.
 *   : average    humidity : 65% humidity.
 *   : average    pressure : 30.4Kpa.
 * =========================================
 * INFO: one observer removed. total :1
 * INFO: station is updating measurements... will notify observers...
 * <<-- Statistics Display Board -->>
 *   : average temperature : 41F degrees.
 *   : average    humidity : 67.5% humidity.
 *   : average    pressure : 29.8Kpa.
 * =========================================
 * INFO: one observer removed. total :0
 * INFO: station is updating measurements... will notify observers...
 * WARNING: this observer is NOT registered!
 * WARNING: this observer is NOT registered!
 */