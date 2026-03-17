const Layout = () => import("@/layout/index.vue");

export default {
  path: "/shm",
  name: "SHM",
  component: Layout,
  redirect: "/shm/monitor",
  meta: {
    icon: "ep:monitor",
    title: "结构健康监测",
    rank: 2
  },
  children: [
    {
      path: "/shm/monitor",
      name: "SHMMonitor",
      component: () => import("@/index.vue"),
      meta: {
        title: "监测数据",
        showLink: true,
        activePath: "/shm/monitor"
      }
    }
  ]
} satisfies RouteConfigsTable;

