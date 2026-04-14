# 第19章：完整 Vibe 项目实战

> "理论只有在实践中才能检验。"

---

## 19.1 项目：在线课程平台

我们将用 TDD + DDD + DSL 三位一体，实现一个在线课程平台的核心功能：

**核心功能**：
- 学员购买课程
- 课程进度追踪
- 证书颁发
- 讲师收益分配

---

## 19.2 Step 1：事件风暴（DDD 战略设计）

```
识别的领域事件（橙色便利贴）：
┌─────────────────────────────────────────────────────┐
│  StudentEnrolled   →  LessonCompleted  →  CourseCompleted  │
│  CoursePurchased   →  ProgressUpdated  →  CertificateIssued│
│  InstructorPaid    ←  RevenueCalculated←  CourseCompleted  │
└─────────────────────────────────────────────────────┘

识别的限界上下文：
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   课程目录上下文   │  │   学习追踪上下文   │  │   支付上下文      │
│                  │  │                  │  │                  │
│ Course           │  │ Enrollment       │  │ Purchase         │
│ Lesson           │  │ Progress         │  │ InstructorPayout │
│ Instructor       │  │ Certificate      │  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
```

---

## 19.3 Step 2：统一语言词汇表

```markdown
# 课程平台统一语言词汇表

## 课程目录上下文

**Course（课程）**：由讲师创建的学习内容集合
- 属性：title, description, price, status
- 状态：DRAFT（草稿）→ PUBLISHED（已发布）→ ARCHIVED（已归档）

**Lesson（课时）**：课程中的一个视频单元
- 属性：title, duration, order_number

**Instructor（讲师）**：创建课程的内容提供者

## 学习追踪上下文

**Enrollment（注册/报名）**：学员和课程的关联
- 由购买行为创建
- 包含学习进度

**Progress（进度）**：学员在某课时的完成状态
- 状态：NOT_STARTED、IN_PROGRESS、COMPLETED

**Certificate（证书）**：课程全部完成后颁发

## 领域事件

**CoursePublished**：课程发布，可以被购买
**StudentEnrolled**：学员报名课程
**LessonCompleted**：学员完成某课时
**CourseCompleted**：学员完成所有课时
**CertificateIssued**：证书颁发
```

---

## 19.3.5 测试基础设施（conftest.py）

在开始写测试前，先搭建测试基础设施：

```python
# tests/conftest.py
"""测试基础设施：内存实现 + 工厂函数"""

import pytest
from decimal import Decimal
from foundations import Money


# ---- 内存仓储实现 ----

class InMemoryCourseRepo:
    def __init__(self):
        self._courses = {}
    
    def save(self, course):
        self._courses[course.id] = course
    
    def find_by_id(self, course_id):
        return self._courses.get(course_id)


class InMemoryEnrollmentRepo:
    def __init__(self):
        self._enrollments = {}
    
    def save(self, enrollment):
        self._enrollments[enrollment.id] = enrollment
    
    def find_by_enrollment(self, enrollment_id):
        return self._enrollments.get(enrollment_id)


class InMemoryPointsRepo:
    def __init__(self):
        self._accounts = {}
    
    def find_by_customer(self, customer_id):
        return self._accounts.get(customer_id)
    
    def save(self, account):
        self._accounts[account.customer_id] = account


class InMemoryCertRepo:
    def __init__(self):
        self._certs = {}
    
    def save(self, cert):
        self._certs[cert.id] = cert
    
    def find_by_enrollment(self, enrollment_id):
        return next((c for c in self._certs.values() 
                     if c.enrollment_id == enrollment_id), None)


class FakeEventBus:
    def __init__(self):
        self.published = []
        self._handlers = {}
    
    def publish(self, event):
        self.published.append(event)
        for handler in self._handlers.get(type(event), []):
            handler(event)
    
    def subscribe(self, event_type):
        def decorator(handler):
            self._handlers.setdefault(event_type, []).append(handler)
            return handler
        return decorator


# ---- 测试工厂函数 ----

def make_draft_course(title="Python 入门", price=None):
    """创建草稿课程"""
    if price is None:
        price = Money(Decimal("99.00"), "CNY")
    return Course.create(title=title, price=price)


def make_lesson(title="第一课", duration_minutes=30):
    """创建课程章节"""
    return Lesson(title=title, duration_minutes=duration_minutes)


def make_confirmed_order(customer_id="student-1", course=None):
    """创建已确认的订单"""
    if course is None:
        course = make_draft_course()
    order = Order.place(
        customer_id=customer_id,
        course_id=course.id,
        price=course.price
    )
    order.confirm()
    return order


# ---- 应用组装 ----

def build_test_application():
    """组装测试用应用（全部使用内存实现）"""
    return {
        "course_repo": InMemoryCourseRepo(),
        "enrollment_repo": InMemoryEnrollmentRepo(),
        "points_repo": InMemoryPointsRepo(),
        "cert_repo": InMemoryCertRepo(),
        "event_bus": FakeEventBus(),
    }
```

---

## 19.4 Step 3：TDD 驱动核心领域（Red 先行）

### 3.1 课程聚合

```python
# tests/unit/catalog/test_course.py

class TestCourseAggregate:
    
    class TestCourseCreation:
        def test_instructor_can_create_draft_course(self):
            instructor = Instructor(id=InstructorId("i1"))
            
            course = Course.create(
                instructor=instructor,
                title="Python 从入门到精通",
                price=Money(Decimal("199"), "CNY")
            )
            
            assert course.status == CourseStatus.DRAFT
            assert course.instructor_id == InstructorId("i1")
        
        def test_new_course_has_no_lessons(self):
            course = make_draft_course()
            assert len(course.lessons) == 0
    
    class TestLessonManagement:
        def test_can_add_lesson_to_draft_course(self):
            course = make_draft_course()
            lesson = Lesson(title="第1课：安装Python", duration_minutes=15)
            
            course.add_lesson(lesson)
            
            assert len(course.lessons) == 1
        
        def test_lessons_are_ordered_by_sequence(self):
            course = make_draft_course()
            course.add_lesson(Lesson(title="第1课", duration_minutes=10))
            course.add_lesson(Lesson(title="第2课", duration_minutes=10))
            
            assert course.lessons[0].order_number == 1
            assert course.lessons[1].order_number == 2
        
        def test_cannot_add_lesson_to_published_course(self):
            course = make_published_course()
            
            with pytest.raises(CourseNotModifiableError):
                course.add_lesson(Lesson(title="Extra", duration_minutes=5))
    
    class TestCoursePublication:
        def test_draft_course_with_lessons_can_be_published(self):
            course = make_draft_course()
            course.add_lesson(make_lesson())
            
            course.publish()
            
            assert course.status == CourseStatus.PUBLISHED
        
        def test_draft_course_without_lessons_cannot_be_published(self):
            course = make_draft_course()
            
            with pytest.raises(CoursePublicationError, match="至少一个课时"):
                course.publish()
        
        def test_publishing_emits_course_published_event(self):
            course = make_draft_course()
            course.add_lesson(make_lesson())
            course.publish()
            
            events = course.pull_events()
            assert any(isinstance(e, CoursePublished) for e in events)
```

### 3.2 注册聚合

```python
# tests/unit/learning/test_enrollment.py

class TestEnrollmentAggregate:
    
    def test_student_can_enroll_in_published_course(self):
        enrollment = Enrollment.create(
            student_id=StudentId("s1"),
            course_id=CourseId("c1"),
            lessons=[LessonRef(id="l1"), LessonRef(id="l2")]
        )
        
        assert enrollment.is_active
        assert enrollment.completion_rate == Decimal("0")
    
    def test_completing_a_lesson_updates_progress(self):
        enrollment = make_enrollment(lesson_count=3)
        
        enrollment.complete_lesson(lesson_id="lesson-1")
        
        progress = enrollment.get_lesson_progress("lesson-1")
        assert progress.status == ProgressStatus.COMPLETED
    
    def test_completion_rate_reflects_completed_lessons(self):
        enrollment = make_enrollment(lesson_count=4)
        enrollment.complete_lesson("l1")
        enrollment.complete_lesson("l2")
        
        assert enrollment.completion_rate == Decimal("0.5")  # 50%
    
    def test_completing_all_lessons_marks_course_as_completed(self):
        enrollment = make_enrollment(lesson_count=2)
        enrollment.complete_lesson("l1")
        enrollment.complete_lesson("l2")
        
        assert enrollment.is_completed
    
    def test_completing_course_emits_course_completed_event(self):
        enrollment = make_enrollment(lesson_count=1)
        enrollment.complete_lesson("l1")
        
        events = enrollment.pull_events()
        assert any(isinstance(e, CourseCompleted) for e in events)
    
    def test_course_completed_event_triggers_certificate_issuance(self):
        # 模拟 Saga：CourseCompleted → CertificateIssued
        enrollment = make_enrollment(lesson_count=1)
        enrollment.complete_lesson("l1")
        
        events = enrollment.pull_events()
        course_completed = next(e for e in events if isinstance(e, CourseCompleted))
        
        # 证书服务监听事件
        cert_service = CertificateService(repo=InMemoryCertRepo())
        cert_service.handle_course_completed(course_completed)
        
        certificate = cert_service.find_by_enrollment(enrollment.id)
        assert certificate is not None
        assert certificate.student_id == enrollment.student_id
```

---

## 19.5 Step 4：实现领域模型（Green）

```python
# src/catalog_context/domain/course.py

from dataclasses import dataclass, field
from enum import Enum
from typing import List
from decimal import Decimal
import uuid

class CourseStatus(Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"

@dataclass(frozen=True)
class CourseId:
    value: str = field(default_factory=lambda: str(uuid.uuid4()))

@dataclass(frozen=True)
class InstructorId:
    value: str

@dataclass
class Lesson:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    duration_minutes: int = 0
    order_number: int = 0

@dataclass
class Course:
    id: CourseId = field(default_factory=CourseId)
    instructor_id: InstructorId = field(default_factory=lambda: InstructorId(""))
    title: str = ""
    price: Money = field(default_factory=lambda: Money(Decimal("0"), "CNY"))
    status: CourseStatus = CourseStatus.DRAFT
    _lessons: List[Lesson] = field(default_factory=list, repr=False)
    _events: List = field(default_factory=list, repr=False)
    
    @classmethod
    def create(cls, instructor: 'Instructor', title: str, price: Money) -> 'Course':
        return cls(
            instructor_id=instructor.id,
            title=title,
            price=price
        )
    
    def add_lesson(self, lesson: Lesson) -> None:
        if self.status != CourseStatus.DRAFT:
            raise CourseNotModifiableError("只有草稿课程可以添加课时")
        lesson.order_number = len(self._lessons) + 1
        self._lessons.append(lesson)
    
    def publish(self) -> None:
        if not self._lessons:
            raise CoursePublicationError("课程发布需要至少一个课时")
        self.status = CourseStatus.PUBLISHED
        self._events.append(CoursePublished(
            course_id=self.id,
            instructor_id=self.instructor_id
        ))
    
    @property
    def lessons(self) -> List[Lesson]:
        return sorted(self._lessons, key=lambda l: l.order_number)
    
    def pull_events(self) -> List:
        events = list(self._events)
        self._events.clear()
        return events
```

---

## 19.6 Step 5：设计 DSL 层

```python
# src/learning_context/dsl/enrollment_dsl.py

class LearningSession:
    """学习会话 DSL——学员视角的操作"""
    
    def __init__(self, student_id: str, enrollment_repo, event_bus):
        self._student_id = StudentId(student_id)
        self._repo = enrollment_repo
        self._bus = event_bus
    
    def studying(self, course_id: str) -> 'LessonProgressDSL':
        enrollment = self._repo.find_by_student_and_course(
            self._student_id, CourseId(course_id)
        )
        return LessonProgressDSL(enrollment, self._repo, self._bus)


class LessonProgressDSL:
    def __init__(self, enrollment, repo, bus):
        self._enrollment = enrollment
        self._repo = repo
        self._bus = bus
    
    def completed_lesson(self, lesson_id: str) -> 'LessonProgressDSL':
        self._enrollment.complete_lesson(lesson_id)
        return self
    
    def and_lesson(self, lesson_id: str) -> 'LessonProgressDSL':
        return self.completed_lesson(lesson_id)
    
    def save(self) -> EnrollmentSummary:
        self._repo.save(self._enrollment)
        events = self._enrollment.pull_events()
        self._bus.publish_all(events)
        return EnrollmentSummary.from_enrollment(self._enrollment)


# 使用：一次完整的学习会话记录
summary = (
    LearningSession(student_id="alice", enrollment_repo=repo, event_bus=bus)
        .studying("python-course")
        .completed_lesson("lesson-1")
        .and_lesson("lesson-2")
        .and_lesson("lesson-3")
        .save()
)

print(f"Alice 完成了 {summary.completion_rate:.0%} 的课程内容")
```

---

## 19.7 Step 6：验收测试（双循环外层）

```python
# tests/acceptance/test_course_purchase_and_completion.py

class TestCoursePurchaseAndCompletion:
    """
    用户故事：
    作为一名学员，
    我购买了 Python 课程，
    完成所有课时后，
    我希望获得完成证书。
    """
    
    def test_student_receives_certificate_after_completing_all_lessons(self):
        # 准备：发布的课程
        app = build_test_application()
        course = app.catalog.find_course("python-101")
        
        # 购买课程
        enrollment = app.payments.purchase(
            student_id="alice",
            course_id=course.id
        )
        
        # 学习所有课时
        session = app.learning.session_for("alice")
        for lesson in course.lessons:
            session.completed_lesson(lesson.id)
        session.save()
        
        # 验证：收到证书
        certificate = app.certificates.find_for_student(
            student_id="alice",
            course_id=course.id
        )
        assert certificate is not None
        assert certificate.issued_at is not None
    
    def test_partial_completion_does_not_issue_certificate(self):
        app = build_test_application()
        
        app.payments.purchase(student_id="bob", course_id="python-101")
        
        # 只完成部分课时
        session = app.learning.session_for("bob")
        session.completed_lesson("lesson-1").save()
        
        certificate = app.certificates.find_for_student("bob", "python-101")
        assert certificate is None
```

---

## 19.8 完整项目结构

```
online-course-platform/
│
├── src/
│   ├── shared_kernel/
│   │   ├── money.py
│   │   └── domain_event.py
│   │
│   ├── catalog_context/
│   │   ├── domain/
│   │   │   ├── course.py         # 聚合根
│   │   │   ├── lesson.py
│   │   │   └── events.py
│   │   └── application/
│   │       └── publish_course.py
│   │
│   ├── learning_context/
│   │   ├── domain/
│   │   │   ├── enrollment.py     # 聚合根
│   │   │   ├── progress.py
│   │   │   └── certificate.py
│   │   ├── application/
│   │   │   └── complete_lesson.py
│   │   └── dsl/
│   │       └── learning_session.py  # 内部 DSL
│   │
│   └── payment_context/
│       ├── domain/
│       │   └── purchase.py
│       └── dsl/
│           └── payment_dsl.py
│
├── tests/
│   ├── unit/
│   │   ├── catalog/              # 课程目录单元测试
│   │   ├── learning/             # 学习追踪单元测试
│   │   └── payment/
│   └── acceptance/
│       └── test_course_workflows.py
│
├── rules/
│   └── pricing_rules.dsl        # 外部 DSL：定价规则
│
└── docs/
    └── ubiquitous-language.md   # 统一语言词汇表
```

---

## 19.9 关键设计决策回顾

| 决策 | 选择 | 理由 |
|------|------|------|
| 聚合边界 | Course 不包含 Enrollment | 它们有独立的生命周期 |
| 上下文通信 | 领域事件（异步） | CourseCompleted → CertificateIssued 解耦 |
| DSL 类型 | 内部 DSL（流式接口） | 代码可读性，无需解析器 |
| 测试策略 | Outside-In TDD | 从用户故事到单元测试 |
| 统一语言 | 学员、课时、完成 | 和产品经理说相同的语言 |

---

## 总结

这个完整项目展示了三位一体的实际效果：
- **DDD**：清晰的上下文边界（课程目录/学习追踪/支付）
- **TDD**：从验收测试到单元测试的双循环驱动
- **DSL**：`LearningSession.studying().completed_lesson().save()` 可读

代码不只是代码，而是精确的业务文档。

---

**下一章**：[高阶模式与未来展望](20-advanced-patterns-future.md)
