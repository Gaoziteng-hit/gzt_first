import gym
import traci
import sumolib
import numpy as np
from scipy.spatial import distance
from math import atan2, degrees
from collections import deque
import os

#返回车辆当前方向角
def angle_between(p1, p2, rl_angle):
	xDiff = p2[0] - p1[0]
	yDiff = p2[1] - p1[1]
	#degree()弧度转角度函数，atan2()返回给定的 X, Y 坐标值的反正切值arctan(x)
	angle = degrees(atan2(yDiff, xDiff))
	# Adding the rotation angle of the agent
	angle += rl_angle
	angle = angle % 360
	return angle

#得到a, b坐标间欧式距离（直线距离）
def get_distance(a, b):
	return distance.euclidean(a, b)


class SumoEnv(gym.Env):
	def __init__(self):
		self.name = 'rlagent'
		self.step_length = 0.4#0.4step算一次
		self.acc_history = deque([0, 0], maxlen=2)#双向队列，可以选择从左侧/右侧添加元素，最大长度为2
		self.grid_state_dim = 3
		'''
		状态共计37维:
		ego车辆（横向位置，纵向速度，加速度，距离，纵向位置，5*1）
		以ego为原点取半径内最近的八辆车（横向位置，纵向速度，加速度，距离，4*8）（无纵向位置）
		'''
		self.state_dim = (4*self.grid_state_dim*self.grid_state_dim)+1 # 5 info for the agent, 4 for everybody else 共计37维
		self.pos = (0, 0)
		self.curr_lane = ''
		self.curr_sublane = -1
		self.target_speed = 0
		self.speed = 0
		self.lat_speed = 0
		self.acc = 0
		self.angle = 0
		self.gui = False
		self.numVehicles = 0
		self.vType = 0
		self.lane_ids = []
		self.max_steps = 10000
		self.curr_step = 0
		self.collision = False
		self.done = False


	def start(self, gui=False, numVehicles=30, vType='human', network_conf="networks/highway/sumoconfig.sumo.cfg", network_xml='networks/highway/highway.net.xml'):
		self.gui = gui
		self.numVehicles = numVehicles
		self.vType = vType
		self.network_conf = network_conf
		self.net = sumolib.net.readNet(network_xml)
		self.curr_step = 0
		self.collision = False
		self.done = False

		# Starting sumo
		home = os.getenv("HOME")

		if self.gui:
			#sumoBinary = home + "/gitprograms/sumo/bin/sumo-gui"
			sumoBinary = home + "/下载/sumo/bin/sumo-gui"
		else:
			#sumoBinary = home + "/gitprograms/sumo/bin/sumo"
			sumoBinary = home + "/下载/sumo/bin/sumo"
		sumoCmd = [sumoBinary, "-c", self.network_conf, "--no-step-log", "true", "-W"]
		traci.start(sumoCmd) # traci.start其实就是将sumo的命令行指令以列表形式读取，运行sumo-gui程序，其指令方式与cmd命令行相同


		self.lane_ids = traci.lane.getIDList() # 获取所有车辆 id(形如"edge_lane")

		# Populating the highway
		# 向网络中添加30辆其他车辆
		for i in range(self.numVehicles):
			veh_name = 'vehicle_' + str(i)
			traci.vehicle.add(veh_name, routeID='route_0', typeID=self.vType, departLane='random')
			# Lane change model comes from bit set 100010101010
			# Go here to find out what does it mean
			# https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html#lane_change_mode_0xb6
			#lane_change_model = np.int('100010001010', 2)
			lane_change_model = 256 #设lane change mode=256即AV可以在安全情况下任意变道
			traci.vehicle.setLaneChangeMode(veh_name, lane_change_model)
		# 添加ego车辆
		traci.vehicle.add(self.name, routeID='route_0', typeID='rl')

		# Do some random step to distribute the vehicles
		# 随机分配车辆(30)
		for step in range(self.numVehicles*4):
			traci.simulationStep() # traci.simulationStep()每触发一次运行一秒，直至触发traci.close()关闭仿真

		# Setting the lane change mode to 0 meaning that we disable any autonomous lane change and collision avoidance
		# ego车 lane change mode = 0
		# 将换道模式设置为0意味着车辆禁用任何自动换道和防撞功能
		traci.vehicle.setLaneChangeMode(self.name, 0)

		# Setting up useful parameters
		self.update_params()

	# 执行一步仿真后更新ego车辆目标速度（当前车道上允许的最大速度），速度，加速度，方向角等信息
	def update_params(self):
		# initialize params
		# 获取ego车position和curr_lane
		self.pos = traci.vehicle.getPosition(self.name) # 在__init__中定义了self.name='rlagent', 返回指定车辆在最后一步 [m,m] 内的位置
		self.curr_lane = traci.vehicle.getLaneID(self.name) # curr_lane大致形如："gneE1_0""gneE1_1""gneE1_2"，012表示右中左车道
		if self.curr_lane == '':
			'''
			if we had collission, the agent is being teleported somewhere else. 
			Therefore I will do simulation step until he get teleported back
			如果碰撞，agent会被传送至别的位置，传送完成后继续模拟
			'''
			'''
			assert这个关键字即为断言，当关键字后面的条件为假时，程序自动崩溃并抛出AssertionError的异常。
			作用：可以用assert关键字在程序中置入检查点，当需要确保程序中的某个条件一定为真才能让程序正常工作的话，assert就有用
			'''
			assert self.collision # 在__init__中定义了self.collision = False
			while self.name in traci.simulation.getStartingTeleportIDList() or traci.vehicle.getLaneID(self.name) == '':
				traci.simulationStep() # 进行一步仿真
			self.curr_lane = traci.vehicle.getLaneID(self.name) # 获得更新后ego所处（"edge_lane"）
		# traci.vehicle.getLaneID(self.name)函数得到的应当是一个以_分隔的字符串(该数据应当与ego车辆所处车道相关)，取第0位赋值给curr_cublane
		self.curr_sublane = int(self.curr_lane.split("_")[1]) # curr_sublane更新为ego当前车道

		self.target_speed = traci.vehicle.getAllowedSpeed(self.name) # 返回当前车道上允许的最大速度，与此车辆的速度因子有关，以 m/s 为单位
		self.speed = traci.vehicle.getSpeed(self.name) # 获得ego车速
		self.lat_speed = traci.vehicle.getLateralSpeed(self.name) # 返回ego车辆在最后一步中的横向速度（以 m/s 为单位）
		self.acc = traci.vehicle.getAcceleration(self.name) # 获得ego加速度
		self.acc_history.append(self.acc) # ego当前加速度存入加速度历史
		self.angle = traci.vehicle.getAngle(self.name) # 获得ego角度


	# Get grid like state 获得网格状状态，此处网格指一个3*3的nparray，矩阵的每个元素为一个车辆id（ego+其他8车）
	def get_grid_state(self, threshold_distance=10):
		'''
		Observation is a grid occupancy grid
		观测是一个网格占用的网格
		'''
		agent_lane = self.curr_lane
		agent_pos = self.pos # self.pos = (0, 0)
		edge = self.curr_lane.split("_")[0] # 把curr_lane（字符串）以_分开，取第0个（即第一个_前的数据），edge为ego当前所处路段！！！
		agent_lane_index = self.curr_sublane # 初始化sublane=-1，update_pramas()中更新为ego当前车道
		'''
		此时self.lane_ids为所有车辆 id(形如["edge_lane",...])
		lane_ids为列表（所有车辆的edge和lane），循环元素lane，如果edge在lane中，则将lane加入列表lanes
		列表lanes为和ego处于同一edge的车辆id信息（edge+lane）
		'''
		lanes = [lane for lane in self.lane_ids if edge in lane]
		state = np.zeros([self.grid_state_dim, self.grid_state_dim]) # 创建一个数组，尺寸为3*3（因为共有三个车道）(self.grid_state_dim = 3)（二维）
		# Putting agent 向网格中放入agent
		agent_x, agent_y = 1, agent_lane_index # 初始y = sublane = -1，后续y = ego当前车道
		state[agent_x, agent_y] = -1 # 此处state[x,y]有什么特殊含义么？？？
		# Put other vehicles 向网格中放入其他车辆（30辆）
		for lane in lanes:
			# Get vehicles in the lane
			vehicles = traci.lane.getLastStepVehicleIDs(lane) # 返回给定车道,!!!车道!!!,上最后一步的车辆 ID
			veh_lane = int(lane.split("_")[-1])
			for vehicle in vehicles:
				if vehicle == self.name: # 跳过ego车
					continue
				# Get angle wrt rlagent
				veh_pos = traci.vehicle.getPosition(vehicle) # Returns the position of the named vehicle within the last step [m,m].
				# If too far, continue 设定距离超过10则太远，跳过
				if get_distance(agent_pos, veh_pos) > threshold_distance:
					continue
				rl_angle = traci.vehicle.getAngle(self.name) #返回ego车辆在最后一步中的角度（以度为单位）
				veh_id = vehicle.split("_")[1]
				angle = angle_between(agent_pos, veh_pos, rl_angle)
				# 给state中某坐标变量赋值，state的类型为numpy.ndarray
				# 按照各个车辆相对于ego车的坐标分类赋值（左。右，左上，右上，左下，右下，上，下）ps其人备注中上下为南北

				# Putting on the right
				if angle > 337.5 or angle < 22.5:
					state[agent_x, veh_lane] = veh_id
				# Putting on the right north
				if angle >= 22.5 and angle < 67.5:
					state[agent_x-1,veh_lane] = veh_id
				# Putting on north
				if angle >= 67.5 and angle < 112.5:
					state[agent_x-1, veh_lane] = veh_id
				# Putting on the left north
				if angle >= 112.5 and angle < 157.5:
					state[agent_x-1, veh_lane] = veh_id
				# Putting on the left
				if angle >= 157.5 and angle < 202.5:
					state[agent_x, veh_lane] = veh_id
				# Putting on the left south
				if angle >= 202.5 and angle < 237.5:
					state[agent_x+1, veh_lane] = veh_id
				if angle >= 237.5 and angle < 292.5:
					# Putting on the south
					state[agent_x+1, veh_lane] = veh_id
				# Putting on the right south
				if angle >= 292.5 and angle < 337.5:
					state[agent_x+1, veh_lane] = veh_id
		# Since the 0 lane is the right most one, flip 因为0车道是最右边的，所以翻转（左右翻转）
		state = np.fliplr(state)
		return state

	# 计算加加速度
	def compute_jerk(self):
		return (self.acc_history[1] - self.acc_history[0])/self.step_length

	# 判断ego车辆有无碰撞，有则True，其他均False
	def detect_collision(self):
		collisions = traci.simulation.getCollidingVehiclesIDList()
		if self.name in collisions:
			self.collision = True
			return True
		self.collision = False
		return False

	# 得到一个37个量组成的状态空间
	def get_state(self):
		'''
		Define a state as a vector of vehicles information
		'''
		state = np.zeros(self.state_dim) # 状态空间state为一个1*37的一位向量
		before = 0
		grid_state = self.get_grid_state().flatten() # 把3*3的二维nparray展开成一维(1*9)向量
		for num, vehicle in enumerate(grid_state): # num为1到9，vehicle为grid_state中循环（即循环九个车id）
			if vehicle == 0: # 该处无车，跳过
				continue
			if vehicle == -1: # 该处为ego车
				vehicle_name = self.name
				before = 1
			else:
				vehicle_name = 'vehicle_'+(str(int(vehicle))) # vehicle_name形如"vehicle_edge_lane"
			veh_info = self.get_vehicle_info(vehicle_name)
			idx_init = num*4
			if before and vehicle != -1:
				idx_init += 1
			idx_fin = idx_init + veh_info.shape[0]
			state[idx_init:idx_fin] = veh_info # 把车辆的状态信息存入状态空间state（1*37的一维向量）
		state = np.squeeze(state) # 把state维度从(1'37)压至(37,)
		return state
	
	# 获得状态信息，ego车（横向位置，纵向位置，速度，横向速度，加速度），其他车辆（距离，速度，加速度，横向位置）
	def get_vehicle_info(self, vehicle_name):
		'''
			Method to populate the vector information of a vehicle
		'''
		if vehicle_name == self.name:
			return np.array([self.pos[0], self.pos[1], self.speed, self.lat_speed, self.acc])
		else:
			lat_pos, long_pos = traci.vehicle.getPosition(vehicle_name)
			long_speed = traci.vehicle.getSpeed(vehicle_name)
			acc = traci.vehicle.getAcceleration(vehicle_name)
			dist = get_distance(self.pos, (lat_pos, long_pos))
			return np.array([dist, long_speed, acc, lat_pos])
		
	'''
	Reward设计：
	舒适度Reward = -0.005 * jerk^2
	效率Reward = 0.0005*(2.5*车道Reward + 2.5*速度Reward + 2.0*变道动作Reward)
	碰撞Reward = -100（if撞），+1（if不撞）
	R_tot = R_comf + R_eff + R_safe
	return [R_tot, R_comf, R_eff, R_safe]
	'''
	def compute_reward(self, collision, action):
		'''
			Reward function is made of three elements:
			 - Comfort 
			 - Efficiency
			 - Safety
			 Taken from Ye et al.
		'''
		# Rewards Parameters
		alpha_comf = 0.005
		w_lane = 2.5
		w_speed = 2.5
		w_change = 2.0
		w_eff = 0.0005
		
		# Comfort reward 
		jerk = self.compute_jerk()
		R_comf = -alpha_comf*jerk**2 # 舒适度Reward = -0.005 * jerk^2
		
		#Efficiency reward
		try:
			lane_width = traci.lane.getWidth(traci.vehicle.getLaneID(self.name)) # 返回车道的宽度，以 m 为单位
		except:
			print(traci.vehicle.getLaneID(self.name))
			lane_width = 3.2 # SUMO 的默认车道宽度为 3.2m
		desired_x = self.pos[0] + lane_width*np.cos(self.angle) # ego x坐标+列宽*cos(ego方向角)
		desired_y = self.pos[1] + lane_width*np.sin(self.angle) # ego y坐标+列宽*sin(ego方向角)
		# 车道Reward
		R_lane = -(np.abs(self.pos[0] - desired_x) + np.abs(self.pos[1] - desired_y)) #
		# Speed 速度Reward = ego实际速度与目标速度（当前车道上允许的最大速度）差的绝对值
		R_speed = -np.abs(self.speed - self.target_speed)
		# Penalty for changing lane 变道动作惩罚
		if action:
			R_change = -1
		else:
			R_change = 1
		# Eff 效率Reward = 0.0005*(2.5*车道Reward + 2.5*速度Reward + 2.0*变道动作Reward)
		R_eff = w_eff*(w_lane*R_lane + w_speed*R_speed + w_change*R_change)
		
		# Safety Reward 碰撞Reward = -100（if撞），+1（if不撞）
		# Just penalize collision for now
		if collision:
			R_safe = -100
		else:
			R_safe = +1
		
		# total reward
		R_tot = R_comf + R_eff + R_safe
		return [R_tot, R_comf, R_eff, R_safe]
		
	'''
	ego执行变道动作，0.1s完成车道更改
	进行一步仿真 - 判断（ego）有无碰撞 - 计算Reward - 执行一步仿真后更新ego车辆目标速度（当前车道上允许的最大速度），速度，加速度，方向角等信息
	- 计算新的state - curr_step + 1 - 若碰撞则结束 
	- return next_state, reward, done, collision
	'''
	def step(self, action):
		'''
		This will :
		- send action, namely change lane or stay 
		- do a simulation step
		- compute reward
		- update agent params 
		- compute nextstate
		- return nextstate, reward and done
		'''
		# Action legend : 0 stay, 1 change to right, 2 change to left
		# self.curr_sublane == 0/1/2 实际对应ego所处车道为右/中/左
		if self.curr_lane[0] == 'e':
			action = 0
		if action != 0:
			if action == 1: #向右变道
				if self.curr_sublane == 1: # ego在中车道
					traci.vehicle.changeLane(self.name, 0, 0.1) # 强制改变车道到给定索引的车道；如果成功，将在给定的时间内（以 s 为单位）选择车道，此处给定0.1s变道时间
				elif self.curr_sublane == 2: # ego在左车道
					traci.vehicle.changeLane(self.name, 1, 0.1)
			if action == 2: # 向左变道
				if self.curr_sublane == 0: # ego在右车道
					traci.vehicle.changeLane(self.name, 1, 0.1)
				elif self.curr_sublane == 1: # ego在中车道
					traci.vehicle.changeLane(self.name, 2, 0.1)
		# Sim step
		traci.simulationStep()
		# Check collision
		collision = self.detect_collision()
		# Compute Reward 
		reward = self.compute_reward(collision, action)
		# Update agent params 
		self.update_params()
		# State 
		next_state = self.get_state()
		# Update curr state
		self.curr_step += 1
		'''
		if self.curr_step > self.max_steps:
			done = True
			self.curr_step = 0
		else:
			done = False
		'''
		# Return
		done = collision
		return next_state, reward, done, collision

	# 重绘环境的一帧。默认模式一般比较友好，如弹出一个窗口
	def render(self, mode='human', close=False):
		pass

	# 重置环境的状态，返回观察
	def reset(self, gui=False, numVehicles=20, vType='human'):
		self.start(gui, numVehicles, vType)
		return self.get_state()

	# 关闭环境，并清除内存
	def close(self):
		traci.close(False)
