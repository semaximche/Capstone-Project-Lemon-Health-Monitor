import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
    index("routes/home.tsx"),
    route("signin", "routes/auth/signin.tsx"),
    route("register", "routes/auth/register.tsx"),
    route("analysis", "routes/analysis.tsx"),
    route("dashboard", "routes/dashboard.tsx"),
] satisfies RouteConfig;
