import { handler } from "./api";

test("handler", () => {
  expect(handler([1])).toBe("$1.00");
});
