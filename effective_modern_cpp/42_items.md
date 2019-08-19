# 42 ITEMS

1. **item 1**: Understand template type deduction.

2. **item 2**: Understand `auto` type deduction.

3. **item 3**: Understand `decltype`.

4. **item 4**: Know how to view deduced types.

5. **item 5**: Prefer `auto` to explicit type declarations.

6. **item 6**: Use the explicitly typed initializer idiom when `auto` deduces undesired types.

7. **item 7**: Distinguish between `()` and `{}` when creating objects.

8. **item 8**: Prefer nullptr to `0` and `NULL`.

9. **item 9**: Prefer alias declarations to typedef.

10. **item 10**: Prefer scoped `enum`s to unscoped `enum`s.

11. **item 11**: Prefer `delete`d functions to `private` undefined ones.

12. **item 12**: Declare overriding functions `override`.

13. **item 13**: Prefer `const_iterators` to `iterators`.

14. **item 14**: Declare functions `noexcept` if they won't emit exceptions.

15. **item 15**: Use `constepxr` whenever possible.

16. **item 16**: Make `const` member functions thread safe.

17. **item 17**: Understand special member function generation.

18. **item 18**: Use `std::unique_ptr` for exclusive-ownership resource management.

19. **item 19**: Use `std::shared_ptr` for shared-ownership resource management.

20. **item 20**: Use `std::weak_ptr` for `std::shared_ptr`-like pointers that can dangle.

21. **item 21**: Prefer `std::make_unique` and `std::make_shared` to direct use of `new`

22. **item 22**: When using the Pimpl Idiom, define special member functions in the implementation file.

23. **item 23**: Understand `std::move` and `std::forward`.

24. **item 24**: Distinguish universal references from rvalue references.

25. **item 25**: Use `std::move` on rvalue references, `std::forward` on universal references.

26. **item 26**: Avoid overloading on universal references.

27. **item 27**: Familiarize yourself with alternatives to overloading on universal references.

28. **item 28**: Understand reference collapsing.

29. **item 29**: Assume that move operations are not present, not cheap, and not used.

30. **item 30**: Familiarize yourself with perfect forwarding.

31. **item 31**: Avoid default capture modes.

32. **item 32**: Use init capture to move objects into closures.

33. **item 33**: Use `decltype` on `auto&&` parameters to std::forward them.

34. **item 34**: Prefer `lambdas` to `std::bind`.

35. **item 35**: Prefer task-based programming to thread-based.

36. **item 36**: Specify `std::launch::async` if asynchronicity is essential.

37. **item 37**: Make `std::thread`s unjoinable on all paths.

38. **item 38**: Be aware of varying thread handle destructor behavior.

39. **item 39**: Consider void futures for one-shot event.

40. **item 40**: Use `std::atomic` for concurrency, volatile for special memory.

41. **item 41**: Consider pass by value for copyable parameters that are cheap to move and always copied.

42. **item 42**: Consider emplacement instead of insertion.
