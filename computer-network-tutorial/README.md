# 从零到系统理解的计算机网络教程：TCP/IP、HTTP、HTTPS 与 TLS

## 项目简介

本教程旨在帮助学习者从一个最常见、也最值得反复追问的问题出发，系统理解现代网络通信的核心链路：

> 当你在浏览器里输入一个 URL 并按下回车，接下来到底发生了什么？

沿着这条主线，教程会依次串起 URL、DNS、MAC、ARP、IP、路由、NAT、端口、Socket、TCP、UDP、HTTP、HTTPS、TLS，以及浏览器如何把服务器返回的数据真正显示成页面。

本教程的重点不是把所有网络话题一次性铺满，而是先帮你建立一张可靠的“网络地图”：

1. 每个协议位于哪一层
2. 它解决的是什么问题
3. 它依赖谁，又为谁服务
4. 它在真实访问流程里的什么位置出现

整个写作尽量坚持三件事：

- **先看全局，再拆细节**：先理解完整链路，再分别理解各层协议
- **从真实场景出发**：围绕“打开网页”“访问接口”“浏览器发请求”这些实际问题展开
- **面对初学者，但不牺牲准确性**：尽量用自然语言解释原理，同时避免过度简化

当前仓库已经完成总览篇、TCP/IP 基础主线、HTTP、HTTPS/TLS、实践排障篇与附录，可按完整导航顺序系统学习。

---

## 目标受众

- 对计算机网络感兴趣，但总觉得 TCP、IP、HTTP、HTTPS、TLS 之间关系混乱的初学者
- Web 前端、后端、全栈开发者，希望补齐网络基础知识
- 想真正理解“打开一个网页为什么会牵扯这么多协议”的自学者
- 准备学习抓包、接口调试、服务排障、浏览器网络分析的工程师
- 备考计算机课程、面试或希望系统梳理网络知识的学习者

---

## 章节导航目录

### 开始之前

- [前言：如何使用本教程](./00-preface.md)
- [总览篇：打开一个 URL 之后发生了什么](./01-overview-what-happens-when-you-open-a-url.md)

> 说明：总览篇是全书入口，用来先建立整体地图，不计入 Part 1 的章节编号；第一部分仍然从第1章开始。

### 第一部分：TCP/IP 与网络基础

| 章节 | 标题 | 主要内容 |
|------|------|----------|
| 第1章 | [为什么要分层](./part1-tcpip/01-why-layering.md) | 网络分层的价值、职责边界、为什么复杂系统需要分层 |
| 第2章 | [OSI 模型与 TCP/IP 模型](./part1-tcpip/02-osi-vs-tcpip.md) | 理论分层与工程分层、常见协议如何映射到不同层 |
| 第3章 | [封装与解封装](./part1-tcpip/03-encapsulation-and-decapsulation.md) | 数据如何逐层加头、逐层拆解，以及每层看到的“数据”是什么 |
| 第4章 | [MAC、IP 与 ARP](./part1-tcpip/04-mac-ip-arp.md) | 本地链路通信、主机寻址与地址解析 |
| 第5章 | [路由与 NAT](./part1-tcpip/05-routing-and-nat.md) | 数据包如何跨网络前进，以及私网如何访问公网 |
| 第6章 | [端口与 Socket](./part1-tcpip/06-ports-and-sockets.md) | 一台机器上的多个进程如何进行网络通信 |
| 第7章 | [DNS 基础](./part1-tcpip/07-dns-basics.md) | 域名如何解析为 IP，递归查询、迭代查询与缓存 |
| 第8章 | [TCP 与 UDP](./part1-tcpip/08-tcp-vs-udp.md) | 两种传输协议的核心差异与适用场景 |
| 第9章 | [TCP 三次握手与可靠性](./part1-tcpip/09-tcp-handshake-and-reliability.md) | 握手、序号、ACK、重传、窗口与可靠传输直觉 |
| 第10章 | [浏览器发出一个包的旅程](./part1-tcpip/10-browser-packet-journey.md) | 把 DNS、路由、IP、TCP、HTTP 等知识串成一条发送路径 |

### 第二部分：HTTP

| 章节 | 标题 | 主要内容 |
|------|------|----------|
| 第11章 | [什么是 HTTP](./part2-http/11-what-is-http.md) | HTTP 的定位、请求-响应模型、资源与无状态性 |
| 第12章 | [URL、URI、方法与状态码](./part2-http/12-url-uri-methods-status-codes.md) | URL/URI 区别、常见方法、状态码与基本语义 |
| 第13章 | [请求头、消息体与 Content-Type](./part2-http/13-headers-body-content-type.md) | 头部与消息体分工、Content-Type 与常见媒体类型 |
| 第14章 | [HTTP/1.1 与 Keep-Alive](./part2-http/14-http1-1-and-keep-alive.md) | 持久连接、减少建连开销，以及 HTTP/1.1 的性能局限 |
| 第15章 | [Cookie、Session 与 Token](./part2-http/15-cookie-session-token.md) | 登录态与会话管理、三者职责与常见组合方式 |
| 第16章 | [HTTP 缓存](./part2-http/16-http-caching.md) | 强缓存、协商缓存、Cache-Control、ETag 与 304 |
| 第17章 | [正向代理、反向代理与 CDN](./part2-http/17-proxy-reverse-proxy-cdn.md) | 代理角色差异、缓存加速、负载分担与部署位置 |
| 第18章 | [同源策略与 CORS](./part2-http/18-same-origin-and-cors.md) | 浏览器同源限制、跨源读取边界与 CORS 授权机制 |
| 第19章 | [HTTP/2](./part2-http/19-http2.md) | 二进制分帧、多路复用、头部压缩与 HTTP 语义不变 |
| 第20章 | [HTTP/3 与 QUIC](./part2-http/20-http3-and-quic.md) | HTTP/3 与 QUIC 的关系、基于 UDP 的现代可靠传输思路 |

### 第三部分：HTTPS 与 TLS

| 章节 | 标题 | 主要内容 |
|------|------|----------|
| 第21章 | [为什么需要 HTTPS](./part3-https-tls/21-why-https.md) | 明文 HTTP 的风险、窃听、篡改与冒充 |
| 第22章 | [TLS 提供了什么](./part3-https-tls/22-what-tls-provides.md) | 机密性、完整性、身份认证与 TLS 的基本定位 |
| 第23章 | [对称加密与非对称加密](./part3-https-tls/23-symmetric-vs-asymmetric-crypto.md) | 两类加密思路、密钥分发问题与 TLS 的组合策略 |
| 第24章 | [证书、CA 与 PKI](./part3-https-tls/24-certificates-ca-pki.md) | 证书结构、CA 角色、PKI 与浏览器验证思路 |
| 第25章 | [域名验证与信任链](./part3-https-tls/25-domain-validation-and-trust-chain.md) | SAN、通配符证书、证书链验证与常见报错来源 |
| 第26章 | [TLS 1.2 握手](./part3-https-tls/26-tls-1-2-handshake.md) | 参数协商、证书验证、密钥交换与握手主流程 |
| 第27章 | [TLS 1.3 握手](./part3-https-tls/27-tls-1-3-handshake.md) | 更少轮次、现代安全设计与 TLS 1.3 主流程 |
| 第28章 | [会话恢复、前向保密与 0-RTT](./part3-https-tls/28-session-resumption-pfs-0rtt.md) | 会话恢复、PFS、0-RTT 的收益与边界 |
| 第29章 | [常见攻击与防御](./part3-https-tls/29-common-attacks-and-defenses.md) | 中间人、降级、证书错误、弱配置与防护思路 |
| 第30章 | [HTTPS 保护不了什么](./part3-https-tls/30-what-https-cannot-protect.md) | HTTPS 的能力边界、传输安全之外的常见风险 |

### 第四部分：实践与排障

| 章节 | 标题 | 主要内容 |
|------|------|----------|
| 第31章 | [curl、OpenSSL、Wireshark 与 DevTools](./part4-practice/31-curl-openssl-wireshark-devtools.md) | 常用工具分工、如何按层次观察请求、证书与数据包 |
| 第32章 | [如何定位 DNS、TCP、TLS、HTTP 问题](./part4-practice/32-how-to-debug-dns-tcp-tls-http.md) | 分层排查路径、常见故障表现与定位思路 |
| 第33章 | [常见误解](./part4-practice/33-common-misconceptions.md) | 回看高频概念误区，澄清 TCP/IP、HTTP、HTTPS、TLS 的职责边界 |
| 第34章 | [学习路径与下一步](./part4-practice/34-learning-path-and-next-steps.md) | 复盘主线、如何继续实践、复述、抓包与深入方向 |

### 附录

| 附录 | 标题 | 主要内容 |
|------|------|----------|
| A1 | [术语表](./appendix/a1-glossary.md) | 高频网络与 Web 术语速查，帮助回看章节时快速定位概念 |
| A2 | [常见端口与状态码](./appendix/a2-common-ports-and-status-codes.md) | 高频端口、HTTP 状态码与排障时的快速参照 |
| A3 | [握手速查表](./appendix/a3-handshake-cheatsheet.md) | TCP 三次握手、TLS 1.2 / TLS 1.3 握手主线速查 |
| A4 | [RFC 与参考资料](./appendix/a4-rfc-and-reference-links.md) | 核心 RFC、浏览器文档与后续深入学习入口 |

---

## 学习路径建议

### 路径一：先建立整体地图（1-2 天）

适合第一次系统接触网络、最想先把关系理顺的学习者：

1. 阅读[前言](./00-preface.md)
2. 认真阅读[总览篇](./01-overview-what-happens-when-you-open-a-url.md)
3. 学习第1章、第7章、第9章、第11章、第21章，先建立 TCP/IP、HTTP、HTTPS 的主线直觉
4. 选读第16章、第24章、第31章，补上缓存、证书和工具观察视角
5. 最后回看第10章，把整条链路再串起来

### 路径二：系统打基础（1-2 周）

适合希望从 TCP/IP 到 HTTPS 按主线系统掌握的学习者：

1. 从总览篇开始建立全局图景
2. 按顺序学习第1-10章，打牢 TCP/IP 基础
3. 按顺序学习第11-20章，系统掌握 HTTP 的语义、状态管理、缓存、代理、CORS、HTTP/2 与 HTTP/3
4. 按顺序学习第21-30章，系统掌握 HTTPS/TLS、证书、信任链、TLS 1.2 / TLS 1.3、会话恢复与安全边界
5. 结合第31-34章与附录 A1-A4 做工具实践、排障复盘和速查巩固

### 路径三：Web 开发者补网络基础（3-5 天）

适合已经会写前后端代码，但网络知识不够扎实的工程师：

1. 先读总览篇，明确整条链路
2. 重点学习第4章、第6章、第7章、第9章、第10章
3. 再重点学习第11章、第13章、第15章、第16章、第17章、第18章、第19章、第20章，建立 Web 请求、状态管理、缓存与跨源直觉
4. 继续学习第21章、第24章、第25章、第27章、第29章、第30章，补齐 HTTPS、证书和安全边界视角
5. 配合第31章、第32章和浏览器开发者工具观察真实请求、状态码、头部、缓存、证书与资源加载过程

---

## 前置要求

学习本教程不需要高深的数学背景，也不要求你先学过操作系统或编译原理，但具备以下常识会更轻松：

- 会基本使用浏览器和网页应用
- 知道“网站运行在服务器上”这一事实
- 对客户端、服务器、域名、IP 这些词有大致印象
- 如果有一点编程经验，会更容易理解 API、请求头、状态码、端口等概念

如果完全没有相关基础，也可以直接开始。本教程会尽量从最直观的问题讲起，再逐步引入术语。

---

## 如何使用本教程

1. **先读总览篇，不要急着背缩写**：先知道 URL、DNS、IP、TCP、TLS、HTTP 在整条链路里的位置。
2. **每学一个概念，都问自己三个问题**：它解决什么问题？它在哪一层？它依赖谁，又为谁服务？
3. **带着真实现象去读**：建议边学边打开浏览器开发者工具，观察请求、响应、状态码、资源加载和缓存行为。
4. **反复回到主线问题**：当你学完某一章后，再问自己“这一步会出现在打开 URL 的哪个阶段”。
5. **多画图、多复述**：如果你能把一条请求从浏览器讲到服务器、再讲回浏览器，说明你已经真正开始理解它了。
6. **不要把不同层的问题混在一起**：IP 负责“送到哪台机器”，TCP 负责“怎样可靠送到”，TLS 负责“怎样更安全”，HTTP 负责“消息是什么意思”。

---

## 教程特色

- **以一个 URL 的完整生命周期为主线**：帮助你把零散术语放回真实访问流程中理解
- **分层视角贯穿全书**：持续强调 TCP/IP、HTTP、HTTPS、TLS 之间的职责边界
- **面向初学者但保持技术准确性**：尽量通俗，但避免“好记却不准”的说法
- **强调工程场景**：不仅讲概念，也关注浏览器、操作系统、路由器、服务器如何协同工作
- **适合开发者补基础**：兼顾入门、复习、面试准备与日常排障思路建立
- **中文编写**：术语尽量统一，表达尽量自然，降低阅读门槛

---

## 许可证

本项目采用 MIT 许可证开源。你可以自由使用、复制、修改和分发本教程内容。

---

*如有建议或发现错误，欢迎反馈。*
