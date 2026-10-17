require_relative "util"

class Order
  def initialize(items)
    @items = items
  end

  def total
    Util.fmt(sum)
  end

  def sum
    @items.sum
  end
end
