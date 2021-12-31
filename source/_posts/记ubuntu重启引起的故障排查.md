---
title: 记ubuntu重启引起的故障排查
toc: true
mathjax: true
top: false
cover: true
date: 2020-11-04 15:25:29
categories: 运维
tags:
- ubuntu
- 故障排查
- linux内核
- grub
---

　　前几天，由于自己服务所在的服务器需要停机维护，运行已有近半年的ubuntu 18.04 LTS终于迎来首次重启，尽管由于预料到长时间的服务器不停机，在关机或者重启阶段会有不可预知的事件发生。但是当服务器真的出现网络无法连接的状况的时候，长时间没有安装系统的我那一刻竟然还有点懵B了，此文主要关注此次故障问题排查和解决记录，作为以后服务器维护 参考之用。

### 网络故障

　　重启之后，首先出现的问题就是网络接口灯直接熄灭，几番周折接上显示器和键盘之后，发现是网卡没有ipv4的地址，反而是ipv6的地址是有的。

```bash
ifconfig -a # 显示所有的网卡，如果网卡无ip，单纯的用ifconfig 是无法显示的所有网卡设备的
# or
ip a
```



　　考虑现今的网络设备ipv6基本属于摆设，所以首先定位的问题是静态ip配置没有生效（此刻持续懵B 3mins，完全忘记如何手工配置 静态ip，某些最基础的操作，还是相当依赖UI），搜索一番得到结果是

```bash
vi /etc/network/interfaces
# change
auto enp1s0 # enp5s0 根据ifconfig 或许实际的网卡编号
iface enp1s0 inet static
		address x.x.x.x
		netmask 255.255.255.0
		gateway x.x.x.x
		dns-nameservers 114.114.114.114 8.8.8.8
		
sudo ip a flush enp1s0
sudo systemctl restart networking.service
```

　　悲剧的是，最后重启 networking.service 并没有发现这个服务，一度陷入僵局，最后发现 Ubuntu 18.04 LTS 开始启用 netplan 组建作为网络管理器，所以这里应该使用netplan来配置静态ip，又是搜索一番

```bash
vi /etc/netplan/50-cloud-init.yaml # 对于yaml文件可是深恶痛绝，奇怪冒号后必须接空格 和 不允许 缩进\t的设定
# change
# This file describes the network interfaces available on your system
# For more information, see netplan(5).
network:
  version: 2
  renderer: networkd
  ethernets:
    enp1s0:
     dhcp4: no
     addresses: [192.168.1.222/24]
     gateway4: 192.168.1.1
     nameservers:
       addresses: [8.8.8.8,8.8.4.4]
       
sudo netplan apply
# or
sudo netplan --dubug apply
```

　　坑爹的是，居然没有 netplan 这个命令，然而却有/etc/netplan/50-cloud-init.yaml这个配置网络的文件，而且里面都是曾经配置完好的文件，应该是上次安装就已经确定下来生成的配置文件，这个就属于很小众的问题了，徜徉 StackOverFlow 和 StackExchange 数小时之后得到答案是：从 Ubuntu 16.04  Upgrade Ubuntu 18.04 会出现 netplan 配置已经安装，但是仍然使用 /etc/network/interfaces 配置ip生效的情况。

```bash
# netplan 安装 
sudo apt install netplan.io # 坑爹的软件包命名
```



但是在这两个文件都存在，同时都配置的情况下，仍然无法使静态ip的配置生效，最主要的是，这个服务器就是原生安装的Ubuntu 18.04 ，并不存在Upgrade 导致这个问题存在的原因。最后无奈，只能通过命令临时生效的静态ip配置，来勉强达到可以上网的目的，想来毕竟服务器24h不关机，临时配置也算可用。唯一的缺点就是，万一重启，就又不得不去机房，现场维护，颇为不便。

```bash
# 临时生效的 ip 网关 dns 配置方法
# 以下所有配置 重启失效
ifconfig enp1s0 x.x.x.x netmask 255.255.255.0
route add default gw x.x.x.x
vi /etc/resolf.conf

```



### nvidia-smi 以及 docker 故障

　　更坑爹是，总算以为完事了的时候，发现nvidia-smi挂了，docker也挂了

```bash
nvidia-smi
NVIDIA-SMI has failed because it couldn’t communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
```

### 原因分析　　

　　结合之前网络问题，推测应该是重启导致的系统层面的问题。考虑到此次维护硬件只有内存的变动，通过命令查看内存是正常运作的，判断应该是 Ubuntu 系统在重启的时候发生的变化，导致这个问题发生。以ubuntu重启，nvidia 失效，作为关键词，终于找到本次故障的最终原因：Ubuntu 18.04 会自动更新linux内核，并在重启的时候自动启动最新的内核



### 故障解决，切换成旧版本内核

　　首先需要确认旧版本内核是否可以解决上述所有问题，最简单的切换就内核的方法是，在 grub 启动菜单里面选择,Advanced options for Ubuntu -> Ubuntu , with linux x.x.x-x-generic ， 终于找到了4.15.0-60-generic 是旧版本内核，并且上述问题全部解决，而4.15.0-122-generic则是产生故障的最新内核版本。

　　那么问题来了，怎么切换内核呢？ 搜索引擎救星又来了

```bash
# 查看目前系统已安装内核
dpkg --get-selections |grep linux-image
linux-image-4.15.0-122-generic			deinstall
linux-image-4.15.0-60-generic			install
linux-image-4.15.0-62-generic			deinstall

# 查看 grub 已经生成 菜单入口名称
grep menuentry /boot/grub/grub.cfg
menuentry 'Ubuntu, with Linux 4.15.0-60-generic' 
...
```

#### Solution 1: 修改grub启动配置

```bash
vi /etc/default/grub
# change
GRUB_DEFAULT=“Advanced options for Ubuntu > Ubuntu, with Linux 4.15.0-60-generic”
# 也可以 用数字标示 0作为第一个菜单
GRUB_DEFAULT = "1> 4" #改成这样

GRUB_TIMEOUT_STYLE=menu # default: hidden
GRUB_TIMEOUT=3 # default: 0

sudo update-grub
sudo reboot
```

#### Solution 2: 删除新内核

```bash
# 注意 无法删除正在使用的内核
sudo apt remove linux-image-xxx-xx-generic
# or
sudo dpkg --purge linux-image-x.x.x-xx-generic
# 安装新内核
sudo apt install linux-headers-x.x.x-x-generic linux-image-x.x.x-x-generic
# 关闭内核自动更新
sudo apt-mark hold linux-image-generic linux-headers-generic
# 开启内核自动更新
sudo apt-mark unhold linux-image-generic linux-headers-generic
```

### Conclusion

　　至此，网络问题，nvidia驱动问题，docker问题都得到圆满解决。回顾过程，心态方面还是稍微不够沉着冷静，发现问题根本原因之前，过分关注表象，诸如 netplan 的配置花费的过多时间，反而问题原因没有深入思考，导致东弄西弄一下，试图用碰运气的方式来解决问题方式终究还是有瓶颈的，或许是碰运气的方式曾经取得的成果对现在行为抉择还是产生的一定的影响，其实在陷入僵局之后的思考基本已经锁定了问题，就算是没有解决问题的彼时，心里对故障的排除已经基本有底了，最后的解决也只是水到渠成而已。



References:

[How to configure static IP address on Ubuntu 18.04 ](https://linuxconfig.org/how-to-configure-static-ip-address-on-ubuntu-18-04-bionic-beaver-linux)

[How to enable netplan on ubuntu server upgraded from 16.04 to 18.04](https://askubuntu.com/questions/1034711/how-to-enable-netplan-on-ubuntu-server-upgraded-from-16-04-to-18-04)

[ubuntu18.04 内核自动更新导致驱动掉了](https://blog.csdn.net/qq_43222384/article/details/90314297)