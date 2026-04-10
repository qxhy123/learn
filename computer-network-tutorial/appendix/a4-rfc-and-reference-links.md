# 附录 A4：RFC 与参考资料

## 使用说明

这一附录的目的不是把你一下子推进规范海洋，而是给出一份更适合作为后续深入入口的参考清单。

对初学者来说，更合理的学习顺序通常是：

1. 先用教程建立整体框架
2. 再根据具体问题回到 RFC 或官方文档查细节
3. 把规范内容和抓包、实验、真实请求对应起来

## 第一部分：核心 RFC 参考

### IP、TCP、UDP 相关

- RFC 791 — Internet Protocol
- RFC 768 — User Datagram Protocol
- RFC 9293 — Transmission Control Protocol (TCP)

学习建议：

- 如果你想深入 IP 和 TCP 的规范语义，可以优先从 TCP 总体规范和基础字段开始看
- 入门阶段不必试图一次读完所有细节，更适合带着问题查

### DNS 相关

- RFC 1034 — Domain Names - Concepts and Facilities
- RFC 1035 — Domain Names - Implementation and Specification

学习建议：

- 想深入理解递归、迭代、记录类型和报文格式时，这两份 RFC 是经典起点

### URI / URL 相关

- RFC 3986 — Uniform Resource Identifier (URI): Generic Syntax

学习建议：

- 当你对 URL、URI、查询参数、片段等语法边界想看更正式定义时，这份 RFC 很重要

### HTTP 相关

HTTP 相关规范在现代已由 HTTP 核心系列文档重新整理。常见入口包括：

- RFC 9110 — HTTP Semantics
- RFC 9111 — HTTP Caching
- RFC 9112 — HTTP/1.1
- RFC 9113 — HTTP/2
- RFC 9114 — HTTP/3

学习建议：

- 如果你已经理解本教程里的 HTTP 基础，下一步最值得读的是语义、缓存和所关心版本对应的规范

### TLS 相关

- RFC 8446 — The Transport Layer Security (TLS) Protocol Version 1.3
- RFC 5246 — The Transport Layer Security (TLS) Protocol Version 1.2

学习建议：

- 读 TLS RFC 时，不建议一开始追求全量吃透
- 更适合带着问题看，例如：握手阶段有哪些核心消息、为什么 TLS 1.3 更快、0-RTT 有什么边界

### 证书与 PKI 相关

- RFC 5280 — Internet X.509 Public Key Infrastructure Certificate and Certificate Revocation List (CRL) Profile

学习建议：

- 当你想正式理解证书字段、信任链、扩展、吊销信息时，这份 RFC 是核心参考

## 第二部分：浏览器与 Web 平台参考

### MDN Web Docs

- https://developer.mozilla.org/

适合查：

- HTTP 头部
- 状态码
- Cookie
- 缓存
- CORS
- 浏览器相关行为

学习建议：

- 对 Web 开发者来说，MDN 往往是比硬啃 RFC 更友好的第一参考资料

### Chrome DevTools 文档

- https://developer.chrome.com/docs/devtools/

适合查：

- Network 面板使用方式
- 性能分析
- 调试网页加载、缓存、请求链路等

## 第三部分：TLS、证书与运维实践参考

### OpenSSL 文档

- https://www.openssl.org/docs/

适合查：

- 证书查看
- TLS 连接测试
- 证书链分析
- 常见命令行用法

### Let's Encrypt 文档

- https://letsencrypt.org/
- https://letsencrypt.org/docs/

适合查：

- 公开网站证书自动化签发
- ACME 基础概念
- 实际证书部署与更新思路

### Mozilla SSL Configuration Generator

- https://ssl-config.mozilla.org/

适合查：

- 常见 Web 服务器的 TLS 推荐配置思路
- 现代兼容性与安全性平衡示例

学习建议：

- 做 HTTPS 配置时，这类实践型资料往往比只看理论更快落地

## 第四部分：抓包与网络分析参考

### Wireshark 文档

- https://www.wireshark.org/docs/

适合查：

- 抓包过滤表达式
- 常见协议分析方法
- TCP、TLS、HTTP 的包级观察方式

### IANA 端口与协议注册

- https://www.iana.org/

适合查：

- 端口号
- 协议编号
- 各类互联网注册信息

## 第五部分：如何使用这些参考资料

### 1. 不要把 RFC 当作小说从头连续读

RFC 更适合：

- 确认定义
- 查字段语义
- 回答具体问题
- 和抓包、实验结果互相印证

### 2. 先看教程主线，再回去查规范

例如：

- 学完 HTTP 缓存后再看 RFC 9111
- 学完 TLS 1.3 握手后再看 RFC 8446
- 学完证书链后再看 RFC 5280

这样会比一上来直接读规范更容易建立理解。

### 3. 优先把“规范语义”对应到“现实现象”

例如：

- 在浏览器里看到 `304`，回去查缓存语义
- 在 OpenSSL 输出里看到证书链，回去查 PKI 规范
- 在抓包里看到 ClientHello / ServerHello，回去对照 TLS RFC

这种学习方式比纯理论阅读更稳固。

## 小结

- RFC 和官方文档是网络学习中非常重要的权威参考，但更适合带着问题查
- HTTP、TLS、DNS、TCP、PKI 都有各自值得深入的核心规范
- MDN、Chrome DevTools、OpenSSL、Wireshark、Mozilla SSL 配置资料则更贴近工程实践
- 最有效的深入方式，不是囫囵吞下所有资料，而是把规范、工具和真实现象连接起来
