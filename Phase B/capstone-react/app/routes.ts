import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
    index("routes/home.tsx"),
    route("signin", "routes/auth/signin.tsx"),
    route("register", "routes/auth/register.tsx"),
    route("analysis", "routes/analysis.tsx"),
    route("analyses", "routes/analyses.tsx"),
    route("dashboard", "routes/dashboard.tsx"),
    route("about", "routes/about.tsx"),
] satisfies RouteConfig;
