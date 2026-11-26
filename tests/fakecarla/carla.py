# tests/fakecarla/carla.py — carla_demo.py 를 CARLA 없이 돌려 보기 위한 아주 작은 가짜 CARLA API
# 11/21 실제 J189 측정값 그대로: 도로가 월드 Y 방향, 차도 x -57.7 ~ -37.1 (20.6 m),
# 대기구역 인도 x < -57.7 (폭 4.4 m), 연석~첫 차로 사이 '차로 아닌 아스팔트' 띠, 횡단보도 y 1.66~4.18
# 차로 중심 (대기 쪽 연석에서): 5.33 ←, 9.05 ←, 12.40 →, 16.00 → (← = 바닥 -x = 월드 +Y)
# 실제 CARLA 동작(렌더링·물리)을 흉내 내지 않는다. 스크립트의 흐름·좌표·타이밍만 확인하는 용도.

import math
from enum import Enum

_HOOKS = []


class LaneType(Enum):
    Any = 0
    Driving = 1
    Sidewalk = 2
    Parking = 3
    Bidirectional = 4
    Shoulder = 5
    Border = 6
    Restricted = 7
    Median = 8


class TrafficLightState(Enum):
    Red = 0
    Yellow = 1
    Green = 2


class Vector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class Location(Vector3D):
    def distance(self, o):
        return math.sqrt((self.x - o.x) ** 2 + (self.y - o.y) ** 2 + (self.z - o.z) ** 2)


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll


class Transform:
    def __init__(self, location=None, rotation=None):
        self.location = location or Location()
        self.rotation = rotation or Rotation()

    def get_forward_vector(self):
        y = math.radians(self.rotation.yaw)
        return Vector3D(math.cos(y), math.sin(y), 0)


class WalkerControl:
    def __init__(self, direction=None, speed=0.0):
        self.direction = direction or Vector3D(1, 0, 0)
        self.speed = speed


class GeoLocation:
    def __init__(self, lat, lon):
        self.latitude, self.longitude = lat, lon


R = 6371008.8
CURB_W, CURB_E = -57.70, -37.10
LANES = [(CURB_W + 5.33, 1, 1), (CURB_W + 9.05, 1, 2), (CURB_W + 12.40, -1, -1), (CURB_W + 16.00, -1, -2)]   # (x 중심, 월드 y 방향, lane_id)


class Waypoint:
    def __init__(self, x, y, lane_type, lane=None):
        self.lane_type = lane_type
        self.road_id, self.section_id = 1, 0
        self.lane_id = lane[2] if lane else 0
        self.lane_width = 3.5
        self._dir = lane[1] if lane else 1
        self.transform = Transform(Location(x, y, 0.0), Rotation(yaw=90.0 * self._dir))

    def previous(self, d):
        l = self.transform.location
        return [Waypoint(l.x, l.y - self._dir * d, self.lane_type, (l.x, self._dir, self.lane_id))]

    def next(self, d):
        l = self.transform.location
        return [Waypoint(l.x, l.y + self._dir * d, self.lane_type, (l.x, self._dir, self.lane_id))]


class Map:
    name = "Carla/Maps/Town10HD_Opt"

    def get_crosswalks(self):
        pts = [(CURB_E, 1.66), (CURB_W, 1.66), (CURB_W, 4.18), (CURB_E, 4.18)]
        out = [Location(x, y, 0) for x, y in pts]
        return out + [Location(pts[0][0], pts[0][1], 0)]

    def get_waypoint(self, loc, project_to_road=True, lane_type=LaneType.Driving):
        x = loc.x
        if project_to_road or lane_type == LaneType.Driving:
            lane = min(LANES, key=lambda L: abs(L[0] - x))
            return Waypoint(lane[0], loc.y, LaneType.Driving, lane)
        if CURB_W <= x <= CURB_E:
            lane = min(LANES, key=lambda L: abs(L[0] - x))
            if abs(lane[0] - x) <= 1.75:
                return Waypoint(lane[0], loc.y, LaneType.Driving, lane)
            return None          # 연석과 차로 사이 빈 아스팔트 (실제 J189 처럼)
        if CURB_W - 4.4 < x < CURB_W or CURB_E < x < CURB_E + 4.0:
            return Waypoint(x, loc.y, LaneType.Sidewalk)
        return None

    def transform_to_geolocation(self, loc):
        return GeoLocation(math.degrees(-loc.y / R), math.degrees(loc.x / R))


class Settings:
    def __init__(self):
        self.synchronous_mode = False
        self.fixed_delta_seconds = None


class BBox:
    def __init__(self, ex, ey, ez):
        self.extent = Vector3D(ex, ey, ez)


class Actor:
    _n = 0

    def __init__(self, world, type_id, tf, parent=None):
        Actor._n += 1
        self.id = Actor._n
        self.world, self.type_id, self.tf, self.parent = world, type_id, tf, parent
        self.is_alive = True
        self.cb = None
        self.attrs = {}
        big = "carlacola" in type_id or "fusorosa" in type_id
        self.bounding_box = BBox(4.0, 1.3, 1.8) if big else BBox(2.35, 1.0, 0.75)
        self.control = None

    def get_location(self):
        if self.parent is not None:
            return self.parent.get_location()
        return self.tf.location

    def get_transform(self):
        return self.tf

    def set_transform(self, tf):
        self.tf = tf

    def set_simulate_physics(self, on):
        pass

    def apply_control(self, c):
        self.control = c

    def listen(self, cb):
        self.cb = cb

    def stop(self):
        self.cb = None

    def destroy(self):
        self.is_alive = False
        if self in self.world.actors:
            self.world.actors.remove(self)


class Image:
    def __init__(self, w, h, frame, ts):
        self.width, self.height, self.frame, self.timestamp = w, h, frame, ts
        self.raw_data = bytes(w * h * 4)


class Gnss:
    def __init__(self, lat, lon, ts):
        self.latitude, self.longitude, self.timestamp = lat, lon, ts


class BP:
    def __init__(self, id_):
        self.id = id_
        self.attrs = {"role_name": "", "image_size_x": "800", "image_size_y": "600", "fov": "90",
                      "sensor_tick": "0", "is_invincible": "false"}

    def has_attribute(self, k):
        return k in self.attrs

    def set_attribute(self, k, v):
        self.attrs[k] = v


class Library:
    def find(self, name):
        return BP(name)

    def filter(self, pat):
        if pat.startswith("walker"):
            return [BP("walker.pedestrian.%04d" % i) for i in range(1, 6)]
        return [BP("vehicle.tesla.model3"), BP("vehicle.audi.tt")]


class TrafficLight:
    def __init__(self, i, group):
        self.id, self.group = i, group
        self.state = TrafficLightState.Red

    def get_group_traffic_lights(self):
        return self.group

    def freeze(self, on):
        pass

    def set_state(self, s):
        self.state = s


class Spectator:
    def __init__(self):
        self.tf = Transform()

    def set_transform(self, tf):
        self.tf = tf

    def get_transform(self):
        return self.tf


class World:
    def __init__(self):
        self.map = Map()
        self.settings = Settings()
        self.actors = []
        self.frame = 0
        self.time = 0.0
        self.spec = Spectator()
        g = []
        g.extend([TrafficLight(100 + i, g) for i in range(4)])
        self.lights = g

    def get_map(self):
        return self.map

    def get_blueprint_library(self):
        return Library()

    def get_settings(self):
        s = Settings()
        s.synchronous_mode, s.fixed_delta_seconds = self.settings.synchronous_mode, self.settings.fixed_delta_seconds
        return s

    def apply_settings(self, s):
        self.settings = s

    def get_spectator(self):
        return self.spec

    def get_actors(self):
        return list(self.actors)

    def get_traffic_lights_from_waypoint(self, wp, dist):
        return self.lights[:2]

    def spawn_actor(self, bp, tf, attach_to=None):
        a = Actor(self, bp.id, tf, attach_to)
        a.attrs = dict(bp.attrs)
        self.actors.append(a)
        return a

    def try_spawn_actor(self, bp, tf):
        return self.spawn_actor(bp, tf)

    def tick(self):
        dt = self.settings.fixed_delta_seconds or 0.05
        self.frame += 1
        self.time += dt
        for a in list(self.actors):
            if a.type_id.startswith("walker") and a.control is not None and a.control.speed > 0:
                d = a.control.direction
                n = math.hypot(d.x, d.y) or 1.0
                l = a.tf.location
                a.tf = Transform(Location(l.x + d.x / n * a.control.speed * dt, l.y + d.y / n * a.control.speed * dt,
                                          l.z), a.tf.rotation)
        for a in list(self.actors):
            if a.cb is None:
                continue
            tick = float(a.attrs.get("sensor_tick", "0") or 0)
            if tick and round(self.time / dt) % max(1, round(tick / dt)) != 0:
                continue
            if a.type_id == "sensor.camera.rgb":
                a.cb(Image(int(a.attrs["image_size_x"]), int(a.attrs["image_size_y"]), self.frame, self.time))
            elif a.type_id == "sensor.other.gnss":
                g = self.map.transform_to_geolocation(a.get_location())
                a.cb(Gnss(g.latitude, g.longitude, self.time))
        for h in _HOOKS:
            h(self)
        return self.frame


class Client:
    def __init__(self, host, port):
        self.world = World()

    def set_timeout(self, t):
        pass

    def get_world(self):
        return self.world

    def get_server_version(self):
        return "0.9.16-fake"
