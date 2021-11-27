#include <functional>
#include <ros/ros.h>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/common.hh>

// ROS
#include <thread>
#include "ros/callback_queue.h"
#include "ros/subscribe_options.h"
#include "std_msgs/Float32.h"
#include "geometry_msgs/Twist.h"
#include "std_msgs/Bool.h"

namespace gazebo
{
class ballLauncherPlugin : public ModelPlugin
{
public:
  ballLauncherPlugin() : ModelPlugin()
  {
  }

  void Load(physics::ModelPtr _parent, sdf::ElementPtr /*_sdf*/)
  {

    ROS_INFO("ball launcher plugin loaded");
    // store the pointer to the model
    this->model = _parent;
    this->launch_ball = false;
    this->initial_launch_vel.linear.x = 0;
    this->initial_launch_vel.linear.y = 0;
    this->initial_launch_vel.linear.z = 0;
     //  // create topic names
     std::string vel_cmd_topicname = "/" + this->model->GetName() + "/vel_cmd";
     std::string activate_launch_topicname = "/" + this->model->GetName() + "/activate_launch";
     if (!ros::isInitialized())
     {
         int argc    = 0;
         char** argv = NULL;
         ros::init(argc, argv, "gazebo_client",
           ros::init_options::NoSigintHandler);
     }
     this->rosNode.reset(new ros::NodeHandle("gazebo_client"));
     ros::SubscribeOptions so = ros::SubscribeOptions::create<geometry_msgs::Twist>(vel_cmd_topicname,1,boost::bind(&ballLauncherPlugin::setVelocityCallback, this, _1),ros::VoidPtr(), &this->rosQueue);
     this->vel_cmd_sub = this->rosNode->subscribe(so);
     ros::SubscribeOptions so1 = ros::SubscribeOptions::create<std_msgs::Bool>(activate_launch_topicname,1,boost::bind(&ballLauncherPlugin::activateLaunchCallback, this, _1),ros::VoidPtr(), &this->rosQueue);
     this->activate_launch_sub = this->rosNode->subscribe(so1);
     this->rosQueueThread = std::thread(std::bind(&ballLauncherPlugin::QueueThread, this));
  }// load
  void setVelocityCallback(const geometry_msgs::TwistConstPtr &_msg)
  {
    ROS_INFO("ball_shooter_plugin: setVelocityCallback");

    this->initial_launch_vel = *_msg;
    this->launch_ball = true;
  }

  void activateLaunchCallback(const std_msgs::BoolConstPtr &_msg)
  {
    ROS_INFO("ball_shooter_plugin: activateLaunchCallback");
    if(this->launch_ball)
    {
      this->launch_ball = false;
    }
    else
    {
      ROS_INFO("Velocity not updated, result might be bad");
    }
    this->model->SetLinearVel({initial_launch_vel.linear.x,initial_launch_vel.linear.y, initial_launch_vel.linear.z });
  }

private:

  void QueueThread()
  {
      static const double timeout = 0.01;

      while (this->rosNode->ok())
      {
          this->rosQueue.callAvailable(ros::WallDuration(timeout));
      }
  }


  private:
   physics::ModelPtr model;
   geometry_msgs::Twist initial_launch_vel;
   std::unique_ptr<ros::NodeHandle> rosNode;
   ros::Subscriber rosSub;
   ros::Subscriber vel_cmd_sub;
   ros::Subscriber activate_launch_sub;
   ros::CallbackQueue rosQueue;
   std::thread rosQueueThread;
   bool launch_ball;





};
GZ_REGISTER_MODEL_PLUGIN(ballLauncherPlugin)
}
