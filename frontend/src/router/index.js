import { createWebHistory, createRouter } from 'vue-router'
import Main from "../views/Main.vue"
import Modify from "../views/Modify.vue"

const routes = [
    {
        path: "/",
        name: "Main",
        component: Main
    },
    {
        path: "/modify",
        name: "Modify",
        component: Modify
    }
]

const router = createRouter({
    history: createWebHistory(),
    routes
});

export default router;