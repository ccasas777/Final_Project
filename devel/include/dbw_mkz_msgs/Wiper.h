// Generated by gencpp from file dbw_mkz_msgs/Wiper.msg
// DO NOT EDIT!


#ifndef DBW_MKZ_MSGS_MESSAGE_WIPER_H
#define DBW_MKZ_MSGS_MESSAGE_WIPER_H


#include <string>
#include <vector>
#include <map>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace dbw_mkz_msgs
{
template <class ContainerAllocator>
struct Wiper_
{
  typedef Wiper_<ContainerAllocator> Type;

  Wiper_()
    : status(0)  {
    }
  Wiper_(const ContainerAllocator& _alloc)
    : status(0)  {
  (void)_alloc;
    }



   typedef uint8_t _status_type;
  _status_type status;



  enum {
    OFF = 0u,
    AUTO_OFF = 1u,
    OFF_MOVING = 2u,
    MANUAL_OFF = 3u,
    MANUAL_ON = 4u,
    MANUAL_LOW = 5u,
    MANUAL_HIGH = 6u,
    MIST_FLICK = 7u,
    WASH = 8u,
    AUTO_LOW = 9u,
    AUTO_HIGH = 10u,
    COURTESYWIPE = 11u,
    AUTO_ADJUST = 12u,
    RESERVED = 13u,
    STALLED = 14u,
    NO_DATA = 15u,
  };


  typedef boost::shared_ptr< ::dbw_mkz_msgs::Wiper_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::dbw_mkz_msgs::Wiper_<ContainerAllocator> const> ConstPtr;

}; // struct Wiper_

typedef ::dbw_mkz_msgs::Wiper_<std::allocator<void> > Wiper;

typedef boost::shared_ptr< ::dbw_mkz_msgs::Wiper > WiperPtr;
typedef boost::shared_ptr< ::dbw_mkz_msgs::Wiper const> WiperConstPtr;

// constants requiring out of line definition

   

   

   

   

   

   

   

   

   

   

   

   

   

   

   

   



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::dbw_mkz_msgs::Wiper_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::dbw_mkz_msgs::Wiper_<ContainerAllocator> >::stream(s, "", v);
return s;
}

} // namespace dbw_mkz_msgs

namespace ros
{
namespace message_traits
{



// BOOLTRAITS {'IsFixedSize': True, 'IsMessage': True, 'HasHeader': False}
// {'geometry_msgs': ['/opt/ros/kinetic/share/geometry_msgs/cmake/../msg'], 'std_msgs': ['/opt/ros/kinetic/share/std_msgs/cmake/../msg'], 'dbw_mkz_msgs': ['/capstone/ros/CarND-Capstone/ros/src/dbw_mkz_msgs/msg']}

// !!!!!!!!!!! ['__class__', '__delattr__', '__dict__', '__doc__', '__eq__', '__format__', '__getattribute__', '__hash__', '__init__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_parsed_fields', 'constants', 'fields', 'full_name', 'has_header', 'header_present', 'names', 'package', 'parsed_fields', 'short_name', 'text', 'types']




template <class ContainerAllocator>
struct IsFixedSize< ::dbw_mkz_msgs::Wiper_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::dbw_mkz_msgs::Wiper_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::dbw_mkz_msgs::Wiper_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::dbw_mkz_msgs::Wiper_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dbw_mkz_msgs::Wiper_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::dbw_mkz_msgs::Wiper_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::dbw_mkz_msgs::Wiper_<ContainerAllocator> >
{
  static const char* value()
  {
    return "7fccb48d5d1df108afaa89f8fc14ce1c";
  }

  static const char* value(const ::dbw_mkz_msgs::Wiper_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x7fccb48d5d1df108ULL;
  static const uint64_t static_value2 = 0xafaa89f8fc14ce1cULL;
};

template<class ContainerAllocator>
struct DataType< ::dbw_mkz_msgs::Wiper_<ContainerAllocator> >
{
  static const char* value()
  {
    return "dbw_mkz_msgs/Wiper";
  }

  static const char* value(const ::dbw_mkz_msgs::Wiper_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::dbw_mkz_msgs::Wiper_<ContainerAllocator> >
{
  static const char* value()
  {
    return "uint8 status\n\
\n\
uint8 OFF=0\n\
uint8 AUTO_OFF=1\n\
uint8 OFF_MOVING=2\n\
uint8 MANUAL_OFF=3\n\
uint8 MANUAL_ON=4\n\
uint8 MANUAL_LOW=5\n\
uint8 MANUAL_HIGH=6\n\
uint8 MIST_FLICK=7\n\
uint8 WASH=8\n\
uint8 AUTO_LOW=9\n\
uint8 AUTO_HIGH=10\n\
uint8 COURTESYWIPE=11\n\
uint8 AUTO_ADJUST=12\n\
uint8 RESERVED=13\n\
uint8 STALLED=14\n\
uint8 NO_DATA=15\n\
";
  }

  static const char* value(const ::dbw_mkz_msgs::Wiper_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::dbw_mkz_msgs::Wiper_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.status);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct Wiper_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::dbw_mkz_msgs::Wiper_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::dbw_mkz_msgs::Wiper_<ContainerAllocator>& v)
  {
    s << indent << "status: ";
    Printer<uint8_t>::stream(s, indent + "  ", v.status);
  }
};

} // namespace message_operations
} // namespace ros

#endif // DBW_MKZ_MSGS_MESSAGE_WIPER_H
