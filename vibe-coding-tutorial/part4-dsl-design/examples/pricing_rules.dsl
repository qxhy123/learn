# 电商定价规则 DSL 示例文件
#
# 本文件展示了第15章外部 DSL 的实际使用格式。
# 可通过 PricingEngine.from_file("pricing_rules.dsl") 加载。

rule "VIP 会员折扣"
when customer.tier = "vip"
then discount 10%
end

rule "大额订单优惠"
when order.total > 500
then discount 5%
end

rule "新用户首单"
when customer.is_new = "true"
then fixed_off 20
end

rule "双十一特惠"
when promotion.name = "double11"
then discount 15%
end

rule "满减活动"
when order.total > 200
then fixed_off 30
end
