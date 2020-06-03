#! /usr/bin/env python
# -*- coding: utf-8 -*-


path bound decider
https://blog.csdn.net/linxigjs/article/details/104801877



"""
https://blog.csdn.net/linxigjs/article/details/103789172


车辆自身状态：通过TrajectoryPoint描述。该结构中包含了车辆的位置，速度，加速度，方向等信息。
当前环境信息：通过Frame描述。前面我们已经提到，Frame中包含了一次Planning计算循环中的所有信息。
从这个定义中可以看到，这个结构中包含了这些信息：

障碍物的预测信息
车辆底盘信息
大致定位信息
交通灯信息
导航路由信息
相对地图信息

struct LocalView {
  std::shared_ptr<prediction::PredictionObstacles> prediction_obstacles;
  std::shared_ptr<canbus::Chassis> chassis;
  std::shared_ptr<localization::LocalizationEstimate> localization_estimate;
  std::shared_ptr<perception::TrafficLightDetection> traffic_light;
  std::shared_ptr<routing::RoutingResponse> routing;
  std::shared_ptr<relative_map::MapMsg> relative_map;
  std::shared_ptr<PadMessage> pad_msg;
};


Apollo5.0的Planning模块是基于Scenario、Stage、Task这样的层次组织的，
针对不同的场景设计不同的算法细节。
Scenario指一个特定的问题或场景，
Stage指在一个Scenario下的规划方法的粗略步骤，
Task指一个具体的处理方法。
相应地，一个Scenario包含多个Stage，一个Stage包含多个Task。


区分场景
ScenarioManager
ScenarioManager类用来管理各个Scenario的判别和切换，
通过ScenarioManager::Update() 进而调用 
ScenarioManager::ScenarioDispatch() 来改变当前对应的scenario。


Scenario 
再看Scenario类的定义。
在Scenario::Process()中，
通过调用Stage::Process()来处理该stage所包含的task。
当该stage处理完成时，就切换到下一个stage。
只要当前的stage不是空、有意义，
scenario就是“未完成”的状态，从而可以继续执行接下来的Stage。
当前的stage是空，则所有的stage处理完成了，scenario才处理完毕。

scenario:
switch (current_scenario_->scenario_type()) {
      case ScenarioConfig::LANE_FOLLOW:
      case ScenarioConfig::PULL_OVER:
        break;
      case ScenarioConfig::BARE_INTERSECTION_UNPROTECTED:
      case ScenarioConfig::EMERGENCY_PULL_OVER:
      case ScenarioConfig::PARK_AND_GO:
      case ScenarioConfig::STOP_SIGN_PROTECTED:
      case ScenarioConfig::STOP_SIGN_UNPROTECTED:
      case ScenarioConfig::TRAFFIC_LIGHT_PROTECTED:
      case ScenarioConfig::TRAFFIC_LIGHT_UNPROTECTED_LEFT_TURN:
      case ScenarioConfig::TRAFFIC_LIGHT_UNPROTECTED_RIGHT_TURN:
      case ScenarioConfig::VALET_PARKING:
      case ScenarioConfig::YIELD_SIGN:

// scenario configs
message ScenarioConfig {
  enum ScenarioType {
    LANE_FOLLOW = 0;  // default scenario

    // intersection involved
    BARE_INTERSECTION_UNPROTECTED = 2;
    STOP_SIGN_PROTECTED = 3;
    STOP_SIGN_UNPROTECTED = 4;
    TRAFFIC_LIGHT_PROTECTED = 5;
    TRAFFIC_LIGHT_UNPROTECTED_LEFT_TURN = 6;
    TRAFFIC_LIGHT_UNPROTECTED_RIGHT_TURN = 7;
    YIELD_SIGN = 8;

    // parking
    PULL_OVER = 9;
    VALET_PARKING = 10;

    EMERGENCY_PULL_OVER = 11;
    EMERGENCY_STOP = 12;

    // misc
    NARROW_STREET_U_TURN = 13;
    PARK_AND_GO = 14;
  }




scenarios文件夹中包含了多种场景，内部的每个文件夹就是一个scenario的定义和解决。
首先看Stage类的定义，主要的处理都在Stage::Process()中（此处是纯虚函数）。
我们上面提到了一个Stage包含一个Task List，是在哪设定的呢？
其实是在apollo/modules/planning/conf/scenario/valet_parking_config.pb.txt中。

scenario::STATUS_UNKNOWN
scenario::STATUS_DONE
scenario::STATUS_PROCESSING




Stage:


enum StageType {
    NO_STAGE = 0;

    LANE_FOLLOW_DEFAULT_STAGE = 1;

    // bare_intersection_unprotected scenario
    BARE_INTERSECTION_UNPROTECTED_APPROACH = 200;
    BARE_INTERSECTION_UNPROTECTED_INTERSECTION_CRUISE = 201;

    // stop_sign_unprotected scenario
    STOP_SIGN_UNPROTECTED_PRE_STOP = 300;
    STOP_SIGN_UNPROTECTED_STOP = 301;
    STOP_SIGN_UNPROTECTED_CREEP = 302;
    STOP_SIGN_UNPROTECTED_INTERSECTION_CRUISE = 303;

    // traffic_light_protected scenario
    TRAFFIC_LIGHT_PROTECTED_APPROACH = 400;
    TRAFFIC_LIGHT_PROTECTED_INTERSECTION_CRUISE = 401;

    // traffic_light_unprotected_left_turn scenario
    TRAFFIC_LIGHT_UNPROTECTED_LEFT_TURN_APPROACH = 410;
    TRAFFIC_LIGHT_UNPROTECTED_LEFT_TURN_CREEP = 411;
    TRAFFIC_LIGHT_UNPROTECTED_LEFT_TURN_INTERSECTION_CRUISE = 412;

    // traffic_light_unprotected_right_turn scenario
    TRAFFIC_LIGHT_UNPROTECTED_RIGHT_TURN_STOP = 420;
    TRAFFIC_LIGHT_UNPROTECTED_RIGHT_TURN_CREEP = 421;
    TRAFFIC_LIGHT_UNPROTECTED_RIGHT_TURN_INTERSECTION_CRUISE = 422;

    // pull_over scenario
    PULL_OVER_APPROACH = 500;
    PULL_OVER_RETRY_APPROACH_PARKING = 501;
    PULL_OVER_RETRY_PARKING = 502;

    // emergency_pull_over scenario
    EMERGENCY_PULL_OVER_SLOW_DOWN = 600;
    EMERGENCY_PULL_OVER_APPROACH = 601;
    EMERGENCY_PULL_OVER_STANDBY = 602;

    // emergency_pull_over scenario
    EMERGENCY_STOP_APPROACH = 610;
    EMERGENCY_STOP_STANDBY = 611;

    // valet parking scenario
    VALET_PARKING_APPROACHING_PARKING_SPOT = 700;
    VALET_PARKING_PARKING = 701;

    // park_and_go scenario
    PARK_AND_GO_CHECK = 800;
    PARK_AND_GO_CRUISE = 801;
    PARK_AND_GO_ADJUST = 802;
    PARK_AND_GO_PRE_CRUISE = 803;

    YIELD_SIGN_APPROACH = 900;
    YIELD_SIGN_CREEP = 901;
  };


Stage::ERROR
Stage::RUNNING
Stage::FINISHED



在apollo/modules/planning/tasks文件夹中，
Task分为4类：deciders，optimizers，rss，smoothers。
task.h定义了Task基类，其中重要的是2个Execute()函数。

Task:
1.ExecuteTaskOnReferenceLine
2.ExecuteTaskOnOpenSpace
参考路径选择
交规决策（信号灯、标志）
路径边界决策
速度决策

LANE_CHOOSE_DECIDER
LANE_BORROW_DECIDER
PATH_BOUNDS_DECIDER
PARKING_SPACE_CHOOSE_DECIDER
TRAFFIC_RULES_DECIDER
RULE_BASED_STOP_DECIDER
SPEED_LIMIT_DECIDER



stage_type: LANE_FOLLOW_DEFAULT_STAGE
  enabled: true
  task_type: LANE_CHANGE_DECIDER
  task_type: PATH_REUSE_DECIDER
  task_type: PATH_LANE_BORROW_DECIDER
  task_type: PATH_BOUNDS_DECIDER
  task_type: PIECEWISE_JERK_PATH_OPTIMIZER
  task_type: PATH_ASSESSMENT_DECIDER
  task_type: PATH_DECIDER
  task_type: RULE_BASED_STOP_DECIDER
  task_type: ST_BOUNDS_DECIDER
  task_type: SPEED_BOUNDS_PRIORI_DECIDER
  task_type: DP_ST_SPEED_OPTIMIZER
  task_type: SPEED_DECIDER
  task_type: SPEED_BOUNDS_FINAL_DECIDER
  # task_type: PIECEWISE_JERK_SPEED_OPTIMIZER
  task_type: PIECEWISE_JERK_NONLINEAR_SPEED_OPTIMIZER
  task_type: DECIDER_RSS



```
Note: Side Pass
While the functionality of side pass still exists, it has now been made universal rather than limiting it to a type of scenario. The side-pass feature is incorporated as part of the path-bounds decider task. You can choose to turn it on or off by properly configuring the path-lane-borrow decider task. For example, if you want the vehicle to be agile, then turn side-pass on for all scenarios; if you feel it not safe to side-pass in intersections, then turn it off for those related scenarios.
```

"""