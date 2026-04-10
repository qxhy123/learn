# 第13章：请求头、消息体与 Content-Type

## 本章目标

- 理解 HTTP 消息里“头部”和“消息体”分别负责什么
- 明白请求头、响应头不是正文，而是描述信息
- 认识 `Content-Type` 的作用，以及常见媒体类型
- 能读懂一条带请求体或响应体的基础 HTTP 消息

## 先讲结论

HTTP 消息通常可以分成两部分：

- **头部**：描述这条消息怎么理解、怎么处理
- **消息体**：真正承载要传输的数据内容

而 `Content-Type` 的核心作用就是告诉接收方：

- 这段消息体到底是什么类型的数据
- 应该按什么方式去解析它

如果没有头部，接收方常常不知道正文是什么意思；如果没有消息体，很多请求和响应就只剩下“说明书”，没有真正的数据内容。

## 核心概念

### 1. 头部负责“描述”，消息体负责“承载”

可以把 HTTP 消息想成“信封 + 信件内容”：

- 头部像信封上的说明
- 消息体像真正要传递的正文

例如在请求里：

- 请求头可以说明客户端接受什么格式、发送的是什么格式、带了哪些认证信息
- 请求体则可能是真正提交的表单、JSON 数据或上传文件

在响应里：

- 响应头可以说明返回结果的数据类型、长度、缓存策略
- 响应体则是 HTML、JSON、图片字节流等实际内容

### 2. 不是所有请求都有消息体

初学者容易以为每个 HTTP 请求都带正文，其实不是。

常见情况是：

- `GET` 请求通常没有消息体
- `POST`、`PUT`、`PATCH` 常常会带消息体
- 响应则很多时候会带消息体，但像 `204 No Content` 这类响应可以没有正文

所以“有没有消息体”要看具体方法和场景，而不是死记一种固定格式。

### 3. `Content-Type` 表示消息体的媒体类型

`Content-Type` 很关键，因为它决定接收方怎么解读正文。

常见例子：

- `text/html`：HTML 页面
- `application/json`：JSON 数据
- `text/plain`：普通文本
- `application/x-www-form-urlencoded`：表单编码
- `multipart/form-data`：文件上传表单
- `image/png`：PNG 图片

例如服务器返回 JSON 时，如果响应头写的是：

`Content-Type: application/json`

浏览器或客户端就会知道：响应体应该按 JSON 来理解，而不是当成 HTML。

### 4. 请求和响应都可能有各自的头部

HTTP 里“头部”不是只有一种。

- **请求头**：客户端发给服务器
- **响应头**：服务器发给客户端

常见请求头：

- `Host`
- `Accept`
- `Authorization`
- `Content-Type`
- `Cookie`

常见响应头：

- `Content-Type`
- `Content-Length`
- `Set-Cookie`
- `Cache-Control`
- `Location`

要注意，很多字段名会出现在请求和响应两边，但语义未必完全相同，需要结合上下文看。

## 示例

### 示例 1：一个带 JSON 请求体的请求

```http
POST /api/users HTTP/1.1
Host: example.com
Content-Type: application/json
Content-Length: 27

{"name":"Alice","age":20}
```

可以这样理解：

- `POST` 表示提交数据
- `Content-Type: application/json` 表示请求体是 JSON
- 空行之前是头部
- 空行之后是消息体

如果服务器知道请求体是 JSON，就可以按 JSON 解析出 `name` 和 `age` 字段。

### 示例 2：一个返回 HTML 的响应

```http
HTTP/1.1 200 OK
Content-Type: text/html; charset=utf-8
Content-Length: 56

<html><body><h1>Hello HTTP</h1></body></html>
```

这里可以这样读：

- 状态码 `200` 表示成功
- `Content-Type` 表示响应体是 HTML
- `charset=utf-8` 表示文本编码方式
- 消息体是浏览器真正要渲染的页面内容

### 示例 3：文件上传为什么常见 `multipart/form-data`

当一个表单既要传文本字段，又要传文件时，通常会使用：

`Content-Type: multipart/form-data`

因为这类数据不再是一个简单字符串，而是由多个部分拼起来，每一部分都可以有自己的内容和说明。

## 常见误区

### 误区 1：`Content-Type` 表示“我想接收什么”

不对。`Content-Type` 主要表示“我现在发送的这段消息体是什么类型”。

如果是客户端想告诉服务器“我希望你返回什么格式”，更常见的是用 `Accept`。

### 误区 2：有头部就一定有消息体

不对。很多请求只有头部没有正文，很多响应也可能没有正文。

### 误区 3：消息体就是字符串

不对。消息体可以是文本、JSON、图片、视频、压缩包、二进制文件等各种数据。

## 小结

- HTTP 消息通常由头部和消息体组成
- 头部负责描述元信息，消息体负责承载真实数据
- `Content-Type` 用来说明消息体的数据类型和解析方式
- 真正会读 HTTP，需要先分清“说明信息”和“正文内容”
